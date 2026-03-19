import copy
import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.distributions.categorical import Categorical

from utils import AverageMeter
from SchedulingModel import build_model_input


# PPO Trainer
class Trainer:
    def __init__(self, model, optimizer_params, training_params):
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.model = model
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.logger = lambda x: print(x)

        self.memory = Memory(gamma=training_params['gamma'], gae_lambda=training_params['lambda'])
        self.model_old = copy.deepcopy(model)
        self.num_epoch = 1

    def train(self, env, num_epoch=1):
        self.num_epoch = num_epoch
        self.memory.clear()
        sample_score = self.sample(env)
        policy_loss, value_loss, entropy, total_loss = self.update()
        self.scheduler.step()
        return sample_score, policy_loss, value_loss, entropy, total_loss, self.model.state_dict()

    def sample(self, env):
        score_am = AverageMeter()
        self.model_old.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.model_old.eval()
        self.model_old.set_decode_type('sampling')

        with torch.no_grad():
            episode = 0
            while episode < self.training_params['episode']:
                remaining = self.training_params['episode'] - episode
                batch_size = min(self.training_params['sample_batch_size'], remaining)
                env.generate_data(batch_size)

                state = env.reset_state()
                state_list, seq_list, machine_list, reward_list, prob_list, value_list = [], [], [], [], [], []
                done = False
                while not done:
                    state, scaler = build_model_input(state, env)
                    state_list.append([np.array(s.cpu()) for s in state])
                    action, prob, value, _ = self.model_old(state)

                    state, reward, done = env.step(action)
                    reward = reward / scaler

                    seq_list.append(np.array(action[0].cpu()))
                    machine_list.append(np.array(action[1].cpu()))
                    reward_list.append(reward.astype(np.float32))
                    prob_list.append(np.array(prob.detach().cpu()))
                    value_list.append(np.array(value.detach().cpu()))

                seqs = np.stack(seq_list, axis=1)
                machines = np.stack(machine_list, axis=1)
                rewards = np.stack(reward_list, axis=1)
                probs = np.stack(prob_list, axis=1)
                values = np.stack(value_list, axis=1)

                # Score
                ###############################################
                score_mean = state.finish_time.max(axis=1).max(axis=1).mean()
                score_am.update(score_mean.item(), batch_size)

                self.memory.push(state_list, {
                    'sequences': seqs,
                    'machines': machines,
                    'rewards': rewards,
                    'probs': probs,
                    'values': values
                })

                episode += batch_size

        self.logger('Epoch {:3d}: Sample Score: {:.4f}'.format(self.num_epoch, score_am.avg))
        return score_am.avg

    def update(self):
        policy_l = AverageMeter()
        value_l = AverageMeter()
        entropy_l = AverageMeter()
        total_loss = AverageMeter()
        self.model.train()
        self.model.set_decode_type('teacher_forcing')

        self.memory.cal_gae()

        for i in range(self.training_params['K_epochs']):
            self.memory.iterate_init(shuffle=True)
            for states, mini_batch in self.memory.iterate_once(self.training_params['mini_batch_size']):
                for key in mini_batch:
                    mini_batch[key] = torch.tensor(mini_batch[key])

                prob_list = []
                entropy_list = []
                value_list = []
                step = 0
                total_step = mini_batch['sequences'].shape[1]
                while step < total_step:
                    state = states[step]
                    state = [torch.tensor(s) for s in state]
                    seq = mini_batch['sequences'][:, step]
                    machine = mini_batch['machines'][:, step]
                    action = (seq, machine)

                    # get new prob and value
                    _, prob, value, action_prob = self.model(state, action=action)
                    dist = Categorical(probs=action_prob.flatten(-2, -1))
                    ent = dist.entropy().mean()
                    entropy_list.append(ent)
                    prob_list.append(prob)
                    value_list.append(value)
                    step += 1

                # [batch, step]
                new_probs = torch.stack(prob_list, dim=1)
                new_values = torch.stack(value_list, dim=1)
                old_probs = mini_batch['probs']
                old_values = mini_batch['values']
                advantages = mini_batch['advantages']
                returns = mini_batch['returns']

                pg_loss = self.policy_loss(new_probs, old_probs, advantages)
                vf_loss = self.value_loss(new_values, old_values, returns)
                entropy = - torch.mean(torch.stack(entropy_list))

                loss = pg_loss + self.training_params['vf_coef'] * vf_loss + self.training_params['entropy_coef'] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                policy_l.update(pg_loss.item())
                value_l.update(vf_loss.item())
                entropy_l.update(entropy.item())
                total_loss.update(loss.item())

            self.logger(
                'Epoch {:3d}: (PPO Train {:3d}/{:3d}) Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}'
                .format(self.num_epoch, i + 1, self.training_params['K_epochs'],
                        policy_l.avg, value_l.avg, entropy_l.avg))

        return policy_l.avg, value_l.avg, entropy_l.avg, total_loss.avg

    def policy_loss(self, new_probs, old_probs, advantages):
        clip_range = self.training_params['clip_range']
        ratio = torch.exp(new_probs.log() - old_probs.log())

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        return - torch.min(surr1, surr2).mean()

    def value_loss(self, new_values, old_values, returns):
        # Value loss with clipping
        clip_range = self.training_params['clip_range']
        # Clipped value loss
        value_pred_clipped = old_values + (new_values - old_values).clamp(-clip_range, clip_range)
        loss_unclipped = (new_values - returns).pow(2)
        loss_clipped = (value_pred_clipped - returns).pow(2)
        vf_loss = torch.max(loss_unclipped, loss_clipped).mean()

        return vf_loss

    def load_checkpoint(self, optimizer_params, scheduler_params, path):
        try:
            self.optimizer.load_state_dict(optimizer_params)
            self.logger('Optimizer loaded from {}'.format(path))
        except KeyError:
            self.logger('Saved Optimizer can not be used and new Optimizer is used instead')

        try:
            self.scheduler.load_state_dict(scheduler_params)
            self.logger('Scheduler loaded from {}'.format(path))
        except KeyError:
            self.logger('Saved Scheduler can not be used and new Scheduler is used instead')

    def set_logger(self, logger):
        self.logger = logger


class Memory:
    def __init__(self, gamma, gae_lambda):
        self.states = None
        self.actions = None
        self.data_map = None
        self._num_instance = None
        self._episode = None

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def clear(self):
        self.states = None
        self.actions = None
        self.data_map = None
        self._num_instance = None
        self._episode = None

    def push(self, state, data_map):
        if self.data_map is None:
            self.states = copy.deepcopy(state)
            self.data_map = copy.deepcopy(data_map)
        else:
            for step in range(len(state)):
                step_state = copy.deepcopy(state[step])
                for i in range(len(step_state)):
                    self.states[step][i] = np.concatenate((self.states[step][i], step_state[i]), axis=0)

            for key in data_map:
                self.data_map[key] = np.concatenate((self.data_map[key], data_map[key]), axis=0)

    def iterate_init(self, shuffle=False):
        if shuffle:
            perm = np.arange(self.data_map['values'].shape[0])
            np.random.shuffle(perm)
            for step in range(len(self.states)):
                for i in range(len(self.states[step])):
                    self.states[step][i] = self.states[step][i][perm]
            for key in self.data_map:
                self.data_map[key] = self.data_map[key][perm]

        self._num_instance = self.data_map['values'].shape[0]
        self._episode = 0

    def iterate_once(self, batch_size):
        while self._episode < self._num_instance:
            yield self.get_batch(batch_size)

    def get_batch(self, batch_size):
        remaining = self._num_instance - self._episode
        cur_batch_size = min(batch_size, remaining)

        state = []
        for step in range(len(self.states)):
            step_state = []
            for i in range(len(self.states[step])):
                step_state.append(self.states[step][i][self._episode:self._episode + cur_batch_size])
            state.append(step_state)

        batch = dict.fromkeys(self.data_map)
        for key in self.data_map:
            batch[key] = self.data_map[key][self._episode:self._episode + cur_batch_size]

        self._episode += cur_batch_size

        return state, batch

    def cal_gae(self):
        if self.data_map is not None:
            # [batch, step]
            rewards = self.data_map['rewards']
            # [batch, step]
            values = self.data_map['values'].squeeze()

            batch_size, steps = rewards.shape

            # [batch, step + 1]
            value_temp = np.concatenate([values, np.zeros((batch_size, 1))], axis=1)
            # [batch, step]
            delta = rewards + self.gamma * value_temp[:, 1:] - values

            advantages = np.zeros_like(rewards)
            gae = np.zeros(batch_size)
            for step in reversed(range(steps)):
                gae = delta[:, step] + self.gamma * self.gae_lambda * gae
                advantages[:, step] = gae

            returns = advantages + values

            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = (advantages - advantages.mean(axis=0, keepdims=True)) \
                            / (advantages.std(axis=0, keepdims=True) + 1e-8)

            self.data_map.update({'advantages': advantages, 'returns': returns})
        else:
            raise ValueError('No Data in Memory')