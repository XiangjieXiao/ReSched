from typing import Union, Optional
import numpy as np
from dataclasses import dataclass
import time
import torch
# from cp_sat import fjsp_sat


from SchedulingGenerator import (
    JobSchedulingProblemGenerator as JSSPGen,
    FlexibleJobShopProblemGenerator as FJSPGen,
    FlexibleFlowShopProblemGenerator as FFSPGen
)
from SD1FJSPGenerator import CaseGenerator as SD1Gen


@dataclass
class State:
    # nodes (dynamic input)
    job_idx: np.ndarray = None
    operation_idx: np.ndarray = None
    position_full: np.ndarray = None
    position: np.ndarray = None
    position_rev: np.ndarray = None
    position_rev_full: np.ndarray = None
    m_AT: np.ndarray = None
    o_AT: np.ndarray = None

    # edge (dynamic input)
    duration: np.ndarray = None

    # graph structure (dynamic input)
    action_mask: np.ndarray = None
    dependency: np.ndarray = None
    o2o_mask: np.ndarray = None
    o2m_mask: np.ndarray = None

    # for makespan calculation (dynamic value, static matrix)
    start_time: np.ndarray = None
    available_time_o: np.ndarray = None
    finish_time: np.ndarray = None
    est_finish_time: np.ndarray = None

    # for reward calculation (dynamic value)
    est_last_makespan: np.ndarray = None

@dataclass
class Solution:
    # solution record
    job_idx: Optional[Union[list, np.ndarray]] = None
    operation_idx: Optional[Union[list, np.ndarray]] = None
    jo_idx: Optional[Union[list, np.ndarray]] = None
    machine_idx: Optional[Union[list, np.ndarray]] = None
    start_time: Optional[Union[list, np.ndarray]] = None
    end_time: Optional[Union[list, np.ndarray]] = None

    def get_data(self):
        return self.job_idx, self.operation_idx, self.machine_idx, self.start_time, self.end_time

    def __len__(self):
        return len(self.job_idx)


@dataclass
class Problem:
    # Static Data
    job_idx: np.ndarray = None
    operation_idx: np.ndarray = None
    duration: np.ndarray = None
    dependency: np.ndarray = None
    connection: np.ndarray = None

    def get_data(self):
        return self.job_idx, self.operation_idx, self.duration, self.dependency, self.connection

    def __len__(self):
        return self.job_idx.shape[0]


class SchedulingProblemEnv:
    def __init__(self):
        # for Env initialization
        self.generator = None

        # for batch data generation
        self.problem = Problem()
        self.solution = Solution()
        self.batch_size = None

        self.state = State()
        self.seed = None
        self.rng = np.random.default_rng()

    def step(self, action):
        step_reward = self._step(action)

        # last step is decided by greedy
        if self.state.action_mask.shape[1] == 1:
            duration = self.state.duration.squeeze(axis=1)
            mask_duration = np.where(duration == 0, np.inf, duration)
            finish_time = self.state.m_AT + mask_duration
            machine_idx = np.argmin(finish_time, axis=1)
            action = (np.zeros(self.batch_size, dtype=int), machine_idx)
            _ = self._step(action)

            self.solution.job_idx = np.array(self.solution.job_idx).T
            self.solution.operation_idx = np.array(self.solution.operation_idx).T
            self.solution.machine_idx = np.array(self.solution.machine_idx).T
            self.solution.start_time = np.array(self.solution.start_time).T
            self.solution.end_time = np.array(self.solution.end_time).T
            done = True
        else:
            done = False

        return self.state, step_reward, done

    def _step(self, action):
        raise NotImplementedError("Method '_step' must be implemented in the subclass.")

    def _makespan(self):
        raise NotImplementedError("Method '_makespan' must be implemented in the subclass.")

    def _with_self_connection(self, connection):
        eye = np.eye(connection.shape[1], dtype=bool)[None, :, :]
        return connection | eye

    def generate_data(self, batch_size):
        self.batch_size = batch_size
        (self.problem.job_idx, self.problem.operation_idx, self.problem.duration,
         self.problem.dependency, self.problem.connection) = self.generator.generate_instances(batch_size)
        self.problem.connection = self._with_self_connection(self.problem.connection)

    def load_data(self, batch_size, job_idx, operation_idx, duration, dependency, connection):
        self.batch_size = batch_size
        self.problem.job_idx = job_idx.copy()
        self.problem.operation_idx = operation_idx.copy()
        self.problem.duration = duration.copy()
        self.problem.dependency = dependency.copy()
        self.problem.connection = self._with_self_connection(connection.copy())

    def get_data(self):
        return self.problem.get_data()

    def reset_state(self):
        job_idx, operation_idx, duration, dependency, connection = self.get_data()
        num_machine = duration.shape[-1]
        num_jo = job_idx.shape[1]

        # shape: (batch_size, num_jo-step)
        self.state.job_idx = job_idx.copy()
        # shape: (batch_size, num_jo-step)
        self.state.operation_idx = operation_idx.copy()
        self.state.position_full = operation_idx.copy()
        self.state.position = operation_idx.copy()
        position_rev = pos_to_rev_pos_numpy(operation_idx)
        self.state.position_rev_full = position_rev.copy()
        self.state.position_rev = position_rev.copy()
        # shape: (batch_size, num_machine)
        self.state.m_AT = np.zeros((self.batch_size, num_machine))
        # shape: (batch_size, num_jo-step)
        self.state.o_AT = np.zeros((self.batch_size, num_jo))

        # shape: (batch_size, num_jo-step)
        self.state.duration = duration.copy()

        # Initialize mask_ for action_mask calculation
        self.state.dependency = dependency.copy()
        # shape: (batch_size, num_jo-step(from), num_jo-step(to))
        self.state.o2o_mask = connection.copy()
        # shape: (batch_size, num_jo-step, num_machines)
        self.state.o2m_mask = duration != 0
        # shape: (batch_size, num_jo-step, num_machines)
        self.state.action_mask = self.state.o2m_mask * (dependency.sum(1) == 0)[..., None]

        # Initialize ST, AT and FT for makespan calculation
        self.state.start_time = np.full((self.batch_size, num_jo, num_machine), -1)
        # shape: (batch_size, num_jo, num_machines)
        self.state.available_time_o = np.zeros((self.batch_size, num_jo))
        # shape: (batch_size, num_jo, num_machines)
        self.state.finish_time = np.full((self.batch_size, num_jo, num_machine), -1)

        # Estimate (lower bound) of finish time & makespan
        self.state.est_finish_time = np.min(np.where(duration == 0, np.inf, duration), axis=-1)

        for bs in range(job_idx.shape[0]):
            ins_job_idx = job_idx[bs]
            for job_id in np.unique(ins_job_idx):
                mask = ins_job_idx == job_id
                self.state.est_finish_time[bs, mask] = np.cumsum(self.state.est_finish_time[bs, mask])
        self.state.est_last_makespan = self.state.est_finish_time.max(axis=1)

        # Initialize solution record
        self.solution.job_idx = []
        self.solution.operation_idx = []
        self.solution.jo_idx = []
        self.solution.machine_idx = []
        self.solution.start_time = []
        self.solution.end_time = []

        return self.state

    def set_seed(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if self.generator is not None:
            self.generator.set_seed(seed)


class JSSPEnv(SchedulingProblemEnv):
    def __init__(self, generator_params=None):
        super().__init__()
        if generator_params is not None:
            self.generator = JSSPGen(**generator_params)

    def _step(self, action):
        seq_idx, machine_idx = action
        if isinstance(seq_idx, torch.Tensor):
            seq_idx = np.array(seq_idx.cpu())
            machine_idx = np.array(machine_idx.cpu())
        batch_idx = np.arange(self.batch_size)
        job_idx = self.state.job_idx[batch_idx, seq_idx]
        operation_idx = self.state.operation_idx[batch_idx, seq_idx]

        # update job, operation, machine idx for solution record
        self.solution.job_idx.append(job_idx)
        self.solution.operation_idx.append(operation_idx)
        self.solution.machine_idx.append(machine_idx)

        # update jo idx for solution record
        # get job&operation idx from sequence index
        # get origin(undeleted) operation index
        job_mask = self.problem.job_idx == job_idx[:, None]
        operation_mask = self.problem.operation_idx == operation_idx[:, None]
        jo_idx = np.argmax((job_mask & operation_mask), axis=-1)
        self.solution.jo_idx.append(jo_idx)

        # update makespan & start, end time
        # update start, end for solution record
        makespan, est_makespan = self._makespan(batch_idx, jo_idx, machine_idx)

        # Reward: Estimation (lower bound) makespan reward
        diff_est_makespan = - (est_makespan - self.state.est_last_makespan)
        self.state.est_last_makespan = est_makespan
        step_reward = diff_est_makespan

        # update current state
        self._rebuild_remaining_state(batch_idx, job_idx)

        # get action space mask
        prepared = self.state.dependency.sum(1) == 0
        self.state.action_mask = self.state.o2m_mask * prepared[:, :, None]

        return step_reward

    def _makespan(self, batch_idx, jo_idx, machine_idx):
        b = np.arange(len(batch_idx))

        dep = self.problem.dependency
        pred_mask = dep[b, :, jo_idx]  # (bs, all_ops)
        succ_mask = dep[b, jo_idx, :]  # (bs, all_ops)

        dur = self.problem.duration[b, jo_idx, machine_idx]
        mat = self.state.m_AT[b, machine_idx]
        oat = self.state.available_time_o[b, jo_idx]

        no_pred = pred_mask.sum(axis=1) == 0  # (bs,)
        st = np.where(no_pred, mat, np.maximum(mat, oat))  # (bs,)
        et = st + dur

        self.state.m_AT[b, machine_idx] = et
        self.state.start_time[b, jo_idx, machine_idx] = st
        self.state.finish_time[b, jo_idx, machine_idx] = et

        succ_exist = succ_mask.sum(axis=1) == 1
        first_succ = np.argmax(succ_mask, axis=1)
        self.state.available_time_o[b[succ_exist], first_succ[succ_exist]] = et[succ_exist]

        job_mat = self.problem.job_idx[b]  # (bs, all_ops)
        op_mat = self.problem.operation_idx[b]  # (bs, all_ops)
        job_id = self.problem.job_idx[b, jo_idx]
        op_id = self.problem.operation_idx[b, jo_idx]
        delta = et - self.state.est_finish_time[b, jo_idx]  # (bs,)

        mask = (job_mat == job_id[:, None]) & (op_mat >= op_id[:, None])  # (bs, all_ops)
        self.state.est_finish_time[b] += mask * delta[:, None]

        real_mk = self.state.finish_time.max(axis=1).max(axis=1)
        est_mk = self.state.est_finish_time.max(axis=1)
        return real_mk, est_mk

    def _rebuild_remaining_state(self, batch_idx, job_idx):
        batch_size, seq_len, num_machine = self.problem.duration.shape

        delete_mask = np.ones(self.problem.duration.shape[:2], dtype=bool)
        del_idx = np.array(self.solution.jo_idx).T
        delete_mask[batch_idx[:, None], del_idx] = False

        decided_len = del_idx.shape[-1]

        # shape: (batch_size, seq_len) --> (batch_size, seq_len - decided_len)
        self.state.job_idx = self.problem.job_idx[delete_mask].reshape(batch_size, seq_len - decided_len)
        self.state.operation_idx = self.problem.operation_idx[delete_mask].reshape(batch_size, seq_len - decided_len)
        job_mask = self.problem.job_idx == job_idx[:, None]
        self.state.position_full[job_mask] -= 1
        self.state.position = self.state.position_full[delete_mask].reshape(batch_size, seq_len - decided_len)
        self.state.position_rev = self.state.position_rev_full[delete_mask].reshape(batch_size, seq_len - decided_len)
        self.state.o_AT = self.state.available_time_o[delete_mask].reshape(batch_size, seq_len - decided_len)

        # shape: (batch_size, seq_len, num_machine) --> (batch_size, seq_len - decided_len, num_machine)
        self.state.duration = self.problem.duration[delete_mask].reshape(batch_size, seq_len - decided_len, num_machine)
        self.state.o2m_mask = self.state.duration != 0

        # shape: (batch_size, seq_len, seq_len) --> (batch_size, seq_len - decided_len, seq_len - decided_len)
        mask_2d = delete_mask[:, :, None] & delete_mask[:, None, :]
        self.state.dependency = self.problem.dependency[mask_2d].reshape(
            batch_size, seq_len - decided_len, seq_len - decided_len)
        self.state.o2o_mask = self.problem.connection[mask_2d].reshape(
            batch_size, seq_len - decided_len, seq_len - decided_len)


# FJSPEnv's "_step", "_makespan" and "_rebuild_remaining_state" are the same as JSSPEnv
class FJSPEnv(JSSPEnv):
    def __init__(self, generator_params=None, sd1=False):
        SchedulingProblemEnv.__init__(self)
        if generator_params is not None:
            if sd1:
                num_jobs = generator_params['num_jobs']
                num_mas = generator_params['num_machines']
                self.generator = SD1Gen(num_jobs, num_mas)
                print("In SD1 dataset, the number of operations is randomly sampled from 0.8*num_machines to 1.2*num_machines.")
                print("In SD1 dataset, the processing time is randomly sampled from 1 to 20.")
            else:
                self.generator = FJSPGen(**generator_params)


# FFSPEnv's "_step", "_makespan" and "_rebuild_remaining_state" are the same as JSSPEnv
class FFSPEnv(JSSPEnv):
    def __init__(self, generator_params=None):
        SchedulingProblemEnv.__init__(self)
        if generator_params is not None:
            self.generator = FFSPGen(**generator_params)


def pos_to_rev_pos_numpy(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos)
    assert pos.ndim == 2
    B, T = pos.shape
    starts = (pos == 0).astype(np.int64)
    seg_id = np.cumsum(starts, axis=1) - 1
    Smax = seg_id.max() + 1
    seg_len = np.zeros((B, Smax), dtype=np.int64)
    rows = np.repeat(np.arange(B), T)
    np.add.at(seg_len, (rows, seg_id.ravel()), 1)
    len_per_pos = seg_len[rows, seg_id.ravel()].reshape(B, T)
    rev_pos = len_per_pos - 1 - pos
    return rev_pos




if __name__ == '__main__':
    # data generator
    params_generator = {
        "num_jobs": 20,
        "max_operations": 4,
        "num_operations": 5,
        "min_operations": 6,
        "num_stages": 5,
        "num_machines": 5,
        "min_processing_time": 1,
        "max_processing_time": 99,
        "min_eligible_ma_per_op": 1,
        "max_eligible_ma_per_op": 5,
    }
    num_instance = 42
    env = FJSPEnv(generator_params=params_generator)
    env.generate_data(num_instance)
    state = env.reset_state()

    d = False
    # select random action from batch mask
    start_time = time.time()
    while not d:
        prepared = env.state.dependency.sum(1) == 0
        # simulate model prediction
        action_seq = np.array(
            [np.random.choice(np.where(prepared[i])[0]) for i in range(num_instance)], dtype=int)
        action_mas = np.array(
            [np.random.choice(np.where(env.state.duration[i, ops_idx] > 0)[0]) for i, ops_idx in enumerate(action_seq)])

        # step
        s, r, d = env.step([action_seq, action_mas])

    print(f"Elapsed time: {time.time() - start_time:.2f} sec")
