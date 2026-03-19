import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-processing data for model input
def build_model_input(state, env):
    # Relative Available Time
    m_at = state.m_AT
    o_at = state.o_AT
    pending_mask = state.action_mask.sum(-1) > 0
    o_at_mask = np.where(pending_mask, o_at, np.inf)
    horizon = np.concatenate((m_at, o_at_mask), axis=-1).min(-1)
    o_at = o_at - horizon[:, None]
    m_at = m_at - horizon[:, None]

    # scaler = env.problem.duration.reshape(env.batch_size, -1).max(-1, keepdims=True)
    job_idx = state.job_idx
    scaler = np.full((job_idx.shape[0], 1), env.problem.duration.max())

    m_at = m_at / scaler
    o_at = o_at / scaler
    duration = state.duration / np.expand_dims(scaler, axis=-1)
    min_duration = np.ma.masked_array(state.duration, mask=~state.o2m_mask).min(-1) / scaler

    dependency_o = state.o2o_mask
    dependency_m = state.o2m_mask

    action_mask = state.action_mask

    mapping = _o2o_acc(state.job_idx, state.operation_idx)

    return (
        torch.tensor(min_duration, dtype=torch.float),
        torch.tensor(duration, dtype=torch.float),
        torch.tensor(o_at, dtype=torch.float),
        torch.tensor(m_at, dtype=torch.float),
        torch.tensor(dependency_o, dtype=torch.bool),
        torch.tensor(dependency_m, dtype=torch.bool),
        torch.tensor(action_mask, dtype=torch.bool),
        mapping
    ), scaler.squeeze()

def _o2o_acc(job_idx, operation_idx):
    bs, all_op = job_idx.shape

    max_job = job_idx.max() + 1
    max_operation = operation_idx.max() + 1

    jo_mapping = np.full((bs, max_job, max_operation), fill_value=False, dtype=bool)
    batch_index = np.arange(bs).reshape(-1, 1).repeat(all_op, axis=1)  # shape: (bs, all_op)

    jo_mapping[batch_index, job_idx, operation_idx] = True
    mapping1 = torch.tensor(jo_mapping, dtype=torch.bool)

    triu_base = np.triu(np.ones((max_operation, max_operation), dtype=bool))
    triu_expanded = triu_base[None, None, :, :]

    triu_mask = np.broadcast_to(triu_expanded, (bs, max_job, max_operation, max_operation))
    # o2o-mask
    mapping2 = torch.tensor(triu_mask, dtype=torch.bool)

    return (mapping1, mapping2)


class Model(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.oat_embedding = nn.Linear(in_features=1,
                                       out_features=model_params['embedding_dim'],
                                       bias=True)
        self.mat_embedding = nn.Linear(in_features=1,
                                       out_features=model_params['embedding_dim'],
                                       bias=True)
        self.duration_embedding = nn.Linear(in_features=1,
                                            out_features=model_params['embedding_dim'],
                                            bias=True)
        self.layers = nn.ModuleList([CrossAttentionBlock(**model_params) for _ in range(model_params['block_num'])])

        self.actor = Actor(num_layers=3, input_dim=model_params['embedding_dim']*3, hidden_dim=64, output_dim=1)

        self.decode_type = 'sampling'

    def forward(self, state, action=None):
        (min_duration, duration, o_at, m_at, dependency_o, dependency_m, action_mask, mapping) = state

        # shape: (batch, jo, embedding)
        pending_mask = action_mask.sum(dim=-1) > 0
        operation_embedding = self.duration_embedding(min_duration.unsqueeze(-1))
        operation_embedding[pending_mask] += self.oat_embedding(o_at.unsqueeze(-1))[pending_mask]

        # shape: (batch, machine, embedding)
        machine_embedding = self.mat_embedding(m_at.unsqueeze(-1))

        # shape: (batch, jo, machine, embedding)
        edge_embedding = self.duration_embedding(duration.unsqueeze(-1))

        # message passing
        for layer in self.layers:
            operation_embedding, machine_embedding = layer(operation_embedding, machine_embedding,
                                                           dependency_o, dependency_m, edge_embedding, mapping)

        # decision-making
        prob = self.decision_making(operation_embedding, machine_embedding, edge_embedding, action_mask)

        if self.decode_type == 'sampling':
            # shape: (batch, jo * machine)
            samples = torch.multinomial(input=prob.view(prob.shape[0], -1),
                                        num_samples=1)
            seq_id = (samples // prob.shape[-1]).squeeze(-1)
            machine_id = (samples % prob.shape[2]).squeeze(-1)
        elif self.decode_type == 'greedy':
            samples = torch.argmax(prob.view([prob.shape[0], -1]), dim=-1, keepdim=True)
            seq_id = (samples // prob.shape[-1]).squeeze(-1)
            machine_id = (samples % prob.shape[2]).squeeze(-1)
        elif self.decode_type == 'teacher_forcing':
            seq_id, machine_id = action
        else:
            raise ValueError('Unknown decode type')

        return [seq_id, machine_id], prob[torch.arange(prob.shape[0]), seq_id, machine_id]

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def decision_making(self, operation_embedding, machine_embedding, edge_embedding, action_mask):
        if action_mask.dim() != 3:
            action_mask = action_mask.flatten(1, 2)
            operation_embedding = operation_embedding.flatten(1, 2)
            edge_embedding = edge_embedding.flatten(1, 2)

        # shape: (batch, jo-step, 1, embedding)
        operation = operation_embedding.unsqueeze(2).repeat(1, 1, machine_embedding.shape[1], 1)
        # shape: (batch, 1, machine, embedding)
        machine = machine_embedding.unsqueeze(1).repeat(1, operation_embedding.shape[1], 1, 1)

        # shape: (batch, jo-step, machine, 2*embedding)
        pair = torch.cat([operation, machine, edge_embedding], dim=-1)
        # shape: (batch * action * machine, 3*embedding)
        pair_embedding = pair[action_mask]

        score = self.actor(pair_embedding)

        score_masked = torch.zeros_like(action_mask, dtype=torch.float)
        score_masked[~action_mask] = float('-inf')
        score_masked[action_mask] = score.squeeze(-1)

        # shape: (batch, jo *  machine)
        prob = F.softmax(score_masked.view([score_masked.shape[0], -1]), dim=-1)
        # shape: (batch, jo, machine)
        prob = prob.view(action_mask.shape)

        return prob


class CrossAttentionBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.head_num = model_params['head_num']
        self.qkv_dim = model_params['qkv_dim']

        self.Wq = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False) for _ in range(2)])
        self.Wk = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False) for _ in range(2)])
        self.Wv = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False) for _ in range(2)])
        self.multi_head_combine = nn.ModuleList(
            [nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim) for _ in range(2)])

        self.addAndNormalization = nn.ModuleList([AddAndNormalizationModule(**model_params) for _ in range(2*2)])
        self.feedForward = nn.ModuleList([FeedForwardModule(**model_params) for _ in range(2)])

    def forward(self, operation, machine, mask_o2o, mask_o2m, duration, mapping):
        # Operation Embedding [batch, operation, embedding]
        # Machine Embedding [batch, machine, embedding]

        # Attention 1: operation --> q, operation --> k, v
        # operations get information from other operations(kv)
        ope = self.attention_block(operation, operation, idx=1, mask=mask_o2o, op_mapping=mapping)

        # Attention 2: machine --> q, operation --> k, v
        # machines get information from operations(kv) and duration
        mac = self.attention_block(machine, operation, idx=0, mask=mask_o2m.transpose(1, 2),
                                   edge_weight=duration.transpose(1, 2),
                                   self_flag=True, edge_in_qk=True, edge_in_v=True)

        return ope, mac

    def attention_block(self, input_q, input_kv, idx, mask, op_mapping=None, edge_weight=None,
                        self_flag=False, edge_in_qk=False, edge_in_v=False):
        q = reshape_by_heads(self.Wq[idx](input_q), head_num=self.head_num)
        k = reshape_by_heads(self.Wk[idx](input_kv), head_num=self.head_num)
        v = reshape_by_heads(self.Wv[idx](input_kv), head_num=self.head_num)
        if edge_weight is not None:
            edge_weight = reshape_by_heads(edge_weight, head_num=self.head_num)
            out_concat = multi_head_attention_with_edge(q, k, v, mask, edge_weight, self_flag=self_flag,
                                                        edge_in_qk=edge_in_qk, edge_in_v=edge_in_v)
        else:
            out_concat = multi_head_attention(q, k, v, mask, op_mapping)
        multi_head_out = self.multi_head_combine[idx](out_concat)

        invalid_mask = (mask.sum(dim=-1, keepdim=True) == 0)
        multi_head_out = torch.where(invalid_mask.expand_as(multi_head_out), input_q, multi_head_out)
        out1 = self.addAndNormalization[idx*2](input_q, multi_head_out, mask)
        out2 = self.feedForward[idx](out1)
        output = self.addAndNormalization[idx*2+1](out1, out2, mask)
        output = torch.where(invalid_mask.expand_as(output), input_q, output)

        return output


class AddAndNormalizationModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = StdNormLayer(model_params['embedding_dim'], affine=True)

    def forward(self, input1, input2, mask):
        # input.shape: (batch, *problem, embedding)
        added = input1 + input2

        orig_shape = added.shape
        batch = orig_shape[0]
        embedding = orig_shape[-1]

        # shape: (batch, problem, embedding)
        added_flat = added.reshape(batch, -1, embedding)

        # shape: (batch, problem, embedding)
        normalized = self.norm(added_flat, mask)

        # shape: (batch, *problem, embedding)
        output = normalized.reshape(orig_shape)

        return output


class StdNormLayer(nn.Module):
    def __init__(self, dim, affine=False):
        super().__init__()
        self.affine = affine
        if affine:
            self.alpha = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x, mask):
        none_connected_mask = mask.sum(dim=-1, keepdim=True) == 0
        valid_x = torch.where(none_connected_mask, torch.zeros_like(x), x)

        valid_count = (~none_connected_mask).sum(dim=(1, 2), keepdim=True)
        mean = valid_x.sum(dim=1, keepdim=True) / valid_count
        var = torch.where(none_connected_mask,
                          torch.zeros_like(x), (x - mean) ** 2).sum(dim=1, keepdim=True) / valid_count
        normalized = (x - mean) / torch.sqrt(torch.clamp(var, min=1e-8))

        output = torch.where(none_connected_mask, x, normalized)

        if self.affine:
            output = self.alpha[None, None, :] * output + self.beta[None, None, :]
        return output


class FeedForwardModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.W1 = nn.Linear(model_params['embedding_dim'], model_params['ff_hidden_dim'])
        self.W2 = nn.Linear(model_params['ff_hidden_dim'], model_params['embedding_dim'])

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


def reshape_by_heads(input_tensor, head_num):
    if input_tensor.dim() < 3:
        raise ValueError("Input tensor must have at least 3 dimensions.")

    if input_tensor.size(-1) == 1:
        # If the last dimension is 1, we can treat it as a single head.
        reshaped = input_tensor.unsqueeze(1)
        return reshaped
    elif input_tensor.size(-1) % head_num != 0:
        raise ValueError("The size of the last dimension must be divisible by head_num.")

    key_dim = input_tensor.size(-1) // head_num
    new_shape = input_tensor.shape[:-1] + (head_num, key_dim)
    reshaped = input_tensor.reshape(*new_shape)

    # shape: (batch, ..., embedding) -> (batch, ..., head_num, key_dim) -> (batch, head_num, ..., key_dim)
    tensor_l = len(new_shape)
    perm_order = [0, tensor_l - 2] + list(range(1, tensor_l - 2)) + [tensor_l - 1]
    transposed = reshaped.permute(*perm_order).contiguous()

    return transposed


def multi_head_attention(q, k, v, mask, mapping=None, rope=True):
    # q shape: (batch, head_num, length_q, key_dim)
    # k,v shape: (batch, head_num, length_kv, key_dim)
    bs, h, length_q, key_dim = q.shape
    _, _, length_kv, _ = k.shape

    op_mapping, triu_mask = mapping
    # op_mapping shape: (batch, num_job, num_op) bool
    _, num_job, num_op = op_mapping.shape

    matrix_mapping = op_mapping.unsqueeze(1).unsqueeze(-1).expand(bs, h, -1, -1, key_dim)
    empty = torch.zeros_like(matrix_mapping, dtype=q.dtype, device=q.device)
    matrix_q = empty.clone().masked_scatter_(matrix_mapping, q.flatten())
    matrix_k = empty.clone().masked_scatter_(matrix_mapping, k.flatten())
    matrix_v = empty.clone().masked_scatter_(matrix_mapping, v.flatten())

    if rope:
        matrix_q, matrix_k = apply_rope_mapping(matrix_q, matrix_k)

    # only focus on job-level attention
    # score shape: (batch, head_num, num_job, num_op, num_op)
    score = torch.matmul(matrix_q, matrix_k.transpose(-2, -1))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    mask = triu_mask.unsqueeze(1) & ~(score==0)
    # score_scaled shape: (batch, head_num, num_job, num_op, num_op)
    score_scaled = score_scaled.masked_fill(~mask, float('-inf'))
    # for None-connected nodes(No self-based cross attention)
    # mask shape: (batch, head_num, num_job, num_op, 1)
    invalid_seq = (mask.sum(dim=-1, keepdim=True) == 0)
    score_scaled = score_scaled.masked_fill(invalid_seq.expand_as(score_scaled), float('-1e9'))

    # shape: (batch, head_num, *length_q, *length_kv)
    weights = nn.Softmax(dim=-1)(score_scaled)
    out = torch.matmul(weights, matrix_v)

    out = out[matrix_mapping].reshape(bs, h, length_q, key_dim)

    # shape: (batch, *length_q, head_num, key_dim)
    tensor_l = out.dim()
    perm_order = [0] + list(range(2, tensor_l - 1)) + [1, tensor_l - 1]
    out_transposed = out.permute(*perm_order)

    # shape: (batch, *length_q, head_num*key_dim)
    out_concat = out_transposed.flatten(-2)

    return out_concat


def apply_rope_mapping(matrix_q, matrix_k):
    """
    matrix_q, matrix_k: (bs, head, num_job, num_op, dk)
    """
    bs, head, num_job, num_op, dk = matrix_q.shape
    device = matrix_q.device

    pe = sinusoidal_position_embedding(bs, head, num_job, num_op, dk, device)

    cos_pos = pe[..., dk//2:] .repeat_interleave(2, dim=-1)
    sin_pos = pe[..., :dk//2].repeat_interleave(2, dim=-1)

    q_even = matrix_q[..., ::2]
    q_odd  = matrix_q[..., 1::2]
    q2 = torch.stack([-q_odd, q_even], dim=-1).reshape_as(matrix_q)

    k_even = matrix_k[..., ::2]
    k_odd  = matrix_k[..., 1::2]
    k2 = torch.stack([-k_odd, k_even], dim=-1).reshape_as(matrix_k)

    q_rot = matrix_q * cos_pos + q2 * sin_pos
    k_rot = matrix_k * cos_pos + k2 * sin_pos

    return q_rot, k_rot


def sinusoidal_position_embedding(bs, head, num_job, num_op, dk, device):
    pos = torch.arange(num_op, device=device).float()  # [0,1,...,num_op-1]

    i = torch.arange(dk//2, device=device).float()    # 0,...,dk/2-1
    theta = 1.0 / (10000 ** (2*i/dk))                  # (dk/2,)
    angles = pos.unsqueeze(1) * theta.unsqueeze(0)

    pe = torch.cat([angles.sin(), angles.cos()], dim=1)  # (num_op, dk)
    pe = pe.view(1, 1, 1, num_op, dk)
    pe = pe.expand(bs, head, num_job, num_op, dk)

    return pe


def multi_head_attention_with_edge(q, k, v, mask, edge_weight, self_flag=False, edge_in_qk=False, edge_in_v=False):
    # q shape: (batch, head_num, length_q, key_dim)
    # k,v shape: (batch, head_num, length_kv, key_dim)
    key_dim = q.size(-1)

    # cross attention
    # shape: (batch, head_num, 1, length_kv, qkv_dim)
    k_expanded = k.unsqueeze(2)
    # shape: (batch, head_num, length_q, 1, qkv_dim)
    q_expanded = q.unsqueeze(-2)
    if edge_in_qk:
        # shape: (batch, head_num, length_q, 1, qkv_dim) --> (batch, head_num, length_q, length_kv, qkv_dim)
        q_with_edge = q_expanded + edge_weight
        # shape: (batch, head_num, 1, length_kv, qkv_dim) --> (batch, head_num, length_q, length_kv, qkv_dim)
        k_with_edge = k_expanded + edge_weight
    else:
        # shape: (batch, head_num, length_q, 1, qkv_dim) --> (batch, head_num, length_q, length_kv, qkv_dim)
        q_with_edge = q_expanded
        # shape: (batch, head_num, 1, length_kv, qkv_dim) --> (batch, head_num, length_q, length_kv, qkv_dim)
        k_with_edge = k_expanded
    self_k, self_v = q, q

    if self_flag:
        # score: (batch, head_num, *length_q, *length_kv)
        score = torch.sum(q_with_edge * k_with_edge, dim=-1)
        # self_score: (batch, head_num, length_q, 1)
        self_score = torch.sum(q * self_k, dim=-1, keepdim=True)

        # score: (batch, head_num, *length_q, *length_kv+1)
        score_add = torch.cat((score, self_score), dim=-1)
        score_scaled = score_add / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
        mask_add = torch.cat((mask,
                              torch.ones(mask.shape[0], mask.shape[1], 1, dtype=torch.bool).to(mask.device)), dim=-1)

        score_scaled = score_scaled.masked_fill(~mask_add.unsqueeze(1), float('-inf'))

        # shape: (batch, head_num, *length_q, *length_kv)
        weights = nn.Softmax(dim=-1)(score_scaled)
        if edge_in_v:
            v_with_edge = v.unsqueeze(2) + edge_weight
            out = torch.sum(weights[..., :-1].unsqueeze(-1) * v_with_edge, dim=-2) + weights[..., -1:] * self_v
        else:
            out = torch.matmul(weights[..., :-1], v) + weights[..., -1:] * self_v
    else:
        # score: (batch, head_num, *length_q, *length_kv)
        score = torch.sum(q_with_edge * k_with_edge, dim=-1)
        # scale
        score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
        score_scaled = score_scaled.masked_fill(~mask.unsqueeze(1), float('-inf'))
        # for None-connected nodes(No self-based cross attention)
        invalid_seq = (mask.sum(dim=-1, keepdim=True) == 0)
        score_scaled = score_scaled.masked_fill(invalid_seq.unsqueeze(1), float('-1e9'))

        # shape: (batch, head_num, *length_q, *length_kv)
        weights = nn.Softmax(dim=-1)(score_scaled)

        if edge_in_v:
            v_with_edge = v.unsqueeze(2) + edge_weight
            out = torch.sum(weights.unsqueeze(-1) * v_with_edge, dim=-2)
        else:
            out = torch.matmul(weights, v)

    # shape: (batch, *length_q, head_num, key_dim)
    tensor_l = out.dim()
    perm_order = [0] + list(range(2, tensor_l - 1)) + [1, tensor_l - 1]
    out_transposed = out.permute(*perm_order)

    # shape: (batch, *length_q, head_num*key_dim)
    out_concat = out_transposed.flatten(-2)

    return out_concat


class Actor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of Actor network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Actor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
