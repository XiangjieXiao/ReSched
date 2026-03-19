import numpy as np
from typing import Optional


class SchedulingProblemGenerator:
    def __init__(
            self,
            num_jobs: int,
            num_machines: int,
            num_operations: int,
            min_processing_time: int,
            max_processing_time: int,
            min_operation: int,
            max_operation: int,
            connection_type: str = 'forward_next'
    ):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = num_operations or self.num_machines
        self.min_processing_time = min_processing_time or 1
        self.max_processing_time = max_processing_time
        self.min_operation = min_operation or self.num_operations
        self.max_operation = max_operation or self.num_operations
        self.connection_type = connection_type

        # for seed setting
        self.seed = None
        self.rng = np.random.default_rng()

    def _generate_duration(self, num_jo):
        raise NotImplementedError("Method '_generate_duration' must be implemented in the subclass.")

    def _generate_dependency(self, operation_idx):
        raise NotImplementedError("Method '_generate_dependency' must be implemented in the subclass.")

    def generate_instances(self, instance_num: int):
        """Generate multiple scheduling instances."""
        if self.max_operation != self.min_operation:
            op_per_job = self.rng.integers(self.min_operation, self.max_operation + 1, size=self.num_jobs)
        else:
            op_per_job = self.num_operations * np.ones(self.num_jobs, dtype=int)

        job_id = np.concatenate([np.full(n_op, i) for i, n_op in enumerate(op_per_job)])
        operation_id = np.concatenate([np.arange(n_op) for n_op in op_per_job])

        durations = []
        for _ in range(instance_num):
            duration = self._generate_duration(operation_id)
            durations.append(duration)

        dependency, connection = self._generate_dependency(operation_id)

        # shape: (instance_num, num_jo)
        job_idx = np.tile(job_id[None, :], (instance_num, 1))
        # shape: (instance_num, num_jo)
        operation_idx = np.tile(operation_id[None, :], (instance_num, 1))
        # shape: (instance_num, num_jo, num_machines)
        durations = np.array(durations)
        # shape: (instance_num, num_jo, num_jo)
        dependencies = np.tile(dependency[None, :, :], (instance_num, 1, 1))
        connections = np.tile(connection[None, :, :], (instance_num, 1, 1))

        return job_idx, operation_idx, durations, dependencies, connections

    def set_seed(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)


class JobSchedulingProblemGenerator(SchedulingProblemGenerator):
    def __init__(
            self,
            num_jobs: int,
            num_machines: int,
            max_processing_time: int,
            num_operations: Optional[int] = None,
            min_processing_time: Optional[int] = None,
            min_operation: Optional[int] = None,
            max_operation: Optional[int] = None,
            **unused_kwargs,
    ):
        super().__init__(
            num_jobs,
            num_machines,
            num_operations,
            min_processing_time,
            max_processing_time,
            min_operation,
            max_operation,
        )

    def _generate_duration(self, operation_idx):
        num_jo = len(operation_idx)
        # each machine should have 1 operation at least
        while True:
            duration = np.zeros((num_jo, self.num_machines), dtype=int)
            machine_idx = self.rng.integers(self.num_machines, size=num_jo)
            machine_duration = self.rng.integers(self.min_processing_time,
                                                 self.max_processing_time + 1, size=num_jo)
            duration[np.arange(num_jo), machine_idx] = machine_duration

            if np.sum(duration, axis=0).min() != 0:
                return duration
            else:
                print("Regenerate instance's duration because some machine has no operation assigned.")

    def _generate_dependency(self, operation_idx):
        num_jo = len(operation_idx)
        dependency = np.zeros((num_jo, num_jo), dtype=bool)
        connection = np.zeros((num_jo, num_jo), dtype=bool)

        for idx in range(num_jo):
            # each job's first operation has no predecessor
            if operation_idx[idx] != 0:
                # each operation's predecessor is the previous operation of the same job
                dep_from = idx - 1
                dep_to = idx
                dependency[dep_from, dep_to] = True

                num2end = self.num_operations - operation_idx[idx]
                connection[dep_from, dep_to:(dep_to + num2end)] = True

        return dependency, connection


class FlexibleJobShopProblemGenerator(SchedulingProblemGenerator):
    def __init__(
            self,
            num_jobs: int,
            num_machines: int,
            max_processing_time: int,
            num_operations: Optional[int] = None,
            min_processing_time: Optional[int] = None,
            min_operation: Optional[int] = None,
            max_operation: Optional[int] = None,
            min_eligible_ma_per_op: Optional[int] = None,
            max_eligible_ma_per_op: Optional[int] = None,
            **unused_kwargs,
    ):
        super().__init__(
            num_jobs,
            num_machines,
            num_operations,
            min_processing_time,
            max_processing_time,
            min_operation,
            max_operation,
        )

        self.min_eligible_ma_per_op = min_eligible_ma_per_op or 1
        self.max_eligible_ma_per_op = max_eligible_ma_per_op or self.num_machines

    def _generate_duration(self, operation_idx):
        num_jo = len(operation_idx)
        while True:
            duration = np.zeros((num_jo, self.num_machines), dtype=int)
            # Determine the number of machines to use
            num_machine = self.rng.integers(self.min_eligible_ma_per_op,
                                            self.max_eligible_ma_per_op + 1, size=num_jo)
            for i, mac_num in zip(np.arange(num_jo), num_machine):
                # Randomly select machines
                machine_idx = self.rng.choice(self.num_machines, mac_num, replace=False)

                # Assign the processing time to the selected machines
                duration[i, machine_idx] = self.rng.integers(self.min_processing_time,
                                                             self.max_processing_time + 1, size=mac_num)

            if np.sum(duration, axis=0).min() != 0:
                return duration
            else:
                print("Regenerate instance's duration because some machine has no operation assigned.")

    def _generate_dependency(self, operation_idx):
        num_jo = len(operation_idx)

        # Operation-level dependencies
        dependency_forward_o = np.zeros((num_jo, num_jo), dtype=bool)

        # Job-level dependencies (i.e., current operation to later operations of the same job)
        dependency_forward_j = np.zeros((num_jo, num_jo), dtype=bool)

        for idx in range(num_jo):
            op_id = operation_idx[idx]
            if op_id == 0:
                job_start = idx
                job_end = job_start + 1
                while job_end < num_jo and operation_idx[job_end] != 0:
                    job_end += 1
            else:
                dep_from = idx - 1
                dep_to = idx

                # Operation dependencies
                dependency_forward_o[dep_from, dep_to] = True
                # Job dependencies
                op_range = job_end - dep_to
                dependency_forward_j[dep_from, dep_to:dep_to + op_range] = True

        dependency = dependency_forward_o
        connection = dependency_forward_j

        return dependency, connection


class FlexibleFlowShopProblemGenerator(SchedulingProblemGenerator):
    def __init__(
            self,
            num_jobs: int,
            num_stages: int,
            num_machines: int,
            min_processing_time: int,
            max_processing_time: int,
            machine_cnt_list: list,
            resource_flexibility: bool,
            machine_flexibility: bool,
            **unused_kwargs,
    ):
        num_machines = num_machines or sum(machine_cnt_list)
        super().__init__(
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_operations=num_stages,
            min_processing_time=min_processing_time,
            max_processing_time=max_processing_time,
            # stage/operation num is equal for each job in Flexible Flow problem
            min_operation=None,
            max_operation=None,
        )
        self.num_stages = num_stages
        self.resource_flexibility = resource_flexibility

        # min/max_eligible_ma_per_stage = 1/num_machines when machine_flexibility is True
        self.machine_flexibility = machine_flexibility
        self.machine_cnt_list = machine_cnt_list

        if machine_flexibility:
            if machine_cnt_list is not None:
                raise ValueError("When machine_flexibility is True, machine_cnt_list must be None "
                                 "because machines are shared across stages.")
        else:
            if machine_cnt_list is None:
                raise ValueError("When machine_flexibility is False, machine_cnt_list must be provided.")
            if len(machine_cnt_list) != self.num_stages:
                raise ValueError(f"Length of machine_cnt_list ({len(machine_cnt_list)}) must match "
                                 f"num_stages ({num_stages}).")
            if sum(machine_cnt_list) != self.num_machines:
                raise ValueError(f"num_machines ({num_machines}) must equal sum(machine_cnt_list) "
                                 f"({sum(machine_cnt_list)}).")

    def _generate_duration(self, stage_idx):
        num_jo = len(stage_idx)
        while True:
            duration = np.zeros((num_jo, self.num_machines), dtype=int)
            for stage_num in range(self.num_stages):
                # Step 1 choose machines for this stage
                if self.machine_flexibility:
                    mac_num = self.rng.integers(1, self.num_machines + 1)
                    machine_idx = self.rng.choice(self.num_machines, mac_num, replace=False)
                else:
                    mac_num = self.machine_cnt_list[stage_num]
                    machine_idx = np.arange(mac_num) + (sum(self.machine_cnt_list[:stage_num]) if stage_num > 0 else 0)

                # Step 2 stage duration (job, machine)
                if self.resource_flexibility:
                    dura = self.rng.integers(self.min_processing_time,
                                             self.max_processing_time + 1, size=(self.num_jobs, mac_num))
                else:
                    dura = self.rng.integers(self.min_processing_time,
                                             self.max_processing_time + 1, size=(self.num_jobs, 1))
                    dura = np.repeat(dura, mac_num, axis=1)

                # Step 3 assign duration to the selected machines for each job in this stage
                d = duration[stage_idx == stage_num, :]
                d[:, machine_idx] = dura
                duration[stage_idx == stage_num, :] = d

            if np.sum(duration, axis=0).min() != 0:
                return duration
            else:
                print("Regenerate instance's duration because some machine has no operation assigned.")

    # Same as FlowShopProblem
    def _generate_dependency(self, stage_idx):
        num_jo = len(stage_idx)
        dependency = np.zeros((num_jo, num_jo), dtype=bool)
        connection = np.zeros((num_jo, num_jo), dtype=bool)

        for idx in range(num_jo):
            # each job's first stage has no predecessor
            if stage_idx[idx] != 0:
                # each stage's predecessor is the previous stage of the same job
                dep_from = idx - 1
                dep_to = idx
                dependency[dep_from, dep_to] = True

                num2end = self.num_stages - stage_idx[idx]
                connection[dep_from, dep_to:(dep_to + num2end)] = True

        return dependency, connection


if __name__ == '__main__':
    # params_dist = {
    #     "num_jobs": 20,
    #     "num_operations": None,
    #     "num_stages": None,
    #     "num_machines": 10,
    #     "min_processing_time": 1,
    #     "max_processing_time": 10,
    #     "min_eligible_ma_per_op": 1,
    #     "max_eligible_ma_per_op": None
    # }
    # gen = JobSchedulingProblemGenerator(**params_dist)
    # gen = FlexibleJobShopProblemGenerator(**params_dist)

    params_dist = {
        'num_jobs': 20,
        'num_machines': 12,
        'num_stages': 3,
        'min_processing_time': 2,
        'max_processing_time': 9,
        'resource_flexibility': True,
        'machine_cnt_list': [4, 4, 4],
        'machine_flexibility': False,
    }
    gen = FlexibleFlowShopProblemGenerator(**params_dist)

    a = gen.generate_instances(1000)
    np.savez('./unrelated_10000_problems_444_job100_2_10.pt.npz', *a)
    b = np.load('./unrelated_10000_problems_444_job100_2_10.pt.npz')
    b1 = b['arr_0']
    a = 1