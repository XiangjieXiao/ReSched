from ortools.sat.python import cp_model
import collections
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from SchedulingGenerator import (
    JobSchedulingProblemGenerator,
    FlexibleJobShopProblemGenerator,
    )
from SD1FJSPGenerator import CaseGenerator as SD1FJSPGenerator


def jssp_sat(job_ids, operation_ids, durations):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    num_machines = durations.shape[-1]

    # Computes horizon dynamically as the sum of all durations.
    horizon = durations.max(-1).sum()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, operation_id, duration in zip(job_ids, operation_ids, durations):
        machine_id = duration.argmax()
        duration = duration.max()
        suffix = '_%i_%i' % (job_id, operation_id)
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)
        interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                            'interval' + suffix)
        all_tasks[job_id, operation_id] = task_type(
            start=start_var, end=end_var, interval=interval_var)
        machine_to_intervals[machine_id].append(interval_var)

    # Create and add disjunctive constraints.
    for machine_id in range(num_machines):
        model.AddNoOverlap(machine_to_intervals[machine_id])

    # Precedences inside a job.
    for job_id, operation_id in zip(job_ids, operation_ids):
        if operation_id == 0:
            continue
        else:
            model.Add(all_tasks[job_id, operation_id].start >= all_tasks[job_id, operation_id - 1].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    last_op_list = []
    unique_job_ids = np.unique(job_ids)
    for job_id in unique_job_ids:
        job_mask = (job_ids == job_id)
        last_op_id = operation_ids[job_mask].max()
        last_op_list.append(all_tasks[job_id, last_op_id].end)
    model.AddMaxEquality(obj_var, last_op_list)
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    # Sets a time limit of 3600 seconds.
    solver.parameters.max_time_in_seconds = 1800.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        j, o, m, st, end = [], [], [], [], []
        print(f'Makespan: {solver.Value(obj_var)}')
        schedule = []  # Store the schedule details here.

        # for job_id, job in enumerate(data):
        #     for op_id, ops in enumerate(job):
        for job_id, op_id, duration in zip(job_ids, operation_ids, durations):
            task = all_tasks[job_id, op_id]
            start_time = solver.Value(task.start)
            end_time = solver.Value(task.end)
            schedule.append((job_id, op_id, start_time, end_time))
            print(f'Job {job_id}, Operation {op_id}, Start {start_time}, End {end_time}')
            j.append(job_id)
            o.append(op_id)
            m.append(duration.argmax())
            st.append(start_time)
            end.append(end_time)

    else:
        print('No solution found.')

    if status == cp_model.OPTIMAL:
        print('Optimal makespan:', solver.ObjectiveValue())
        return [0, solver.ObjectiveValue(), [j, o, m, st, end]]
    elif status == cp_model.FEASIBLE:
        print('Feasible makespan:', solver.ObjectiveValue())
        return [1, solver.ObjectiveValue(), [j, o, m, st, end]]
    else:
        print('Not found any Sol. Return [-1, -1]')
        return [-1, -1]


def fjsp_sat(job_ids, operation_ids, durations):
    """Flexible job shop scheduling problem."""
    # Create the model.
    model = cp_model.CpModel()

    # Sum each operation's max duration as horizon
    num_machines = durations.shape[-1]
    horizon = durations.max(-1).sum()

    # Named tuple to store information about created variables.
    operation_type = collections.namedtuple('operation_type', 'start end interval machine')

    # Creates job intervals and add to the corresponding machine lists.
    all_operations = {}
    machine_to_intervals = collections.defaultdict(list)

    # For each job and operation, create decision variables.
    for job_id, operation_id, duration in zip(job_ids, operation_ids, durations):
        suffix = '_%i_%i' % (job_id, operation_id)

        # Create start and end time variables.
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)

        # Create a machine selection variable for each machine (binary).
        machine_vars = []
        for machine_id in range(num_machines):
            if duration[machine_id] > 0:  # machine can process this operation
                # NewBoolVar is a integer variable with domain [0, 1], can not regard as bool type
                machine_var = model.NewBoolVar(f'machine_{machine_id}_for{suffix}')
                machine_vars.append(machine_var)
            else:
                machine_vars.append(None)  # machine cannot process this operation

        # Ensure exactly one machine is chosen for each operation.
        model.Add(sum(m for m in machine_vars if m is not None) == 1)

        # Create interval variables for each machine and associate with binary machine vars.
        intervals = []
        for machine_id, machine_var in enumerate(machine_vars):
            if machine_var is not None:
                interval_var = model.NewOptionalIntervalVar(
                    start_var, duration[machine_id], end_var,
                    is_present=machine_var, name=f'interval_m{machine_id}{suffix}')
                intervals.append(interval_var)
                machine_to_intervals[machine_id].append(interval_var)

        # Store operation-related variables.
        all_operations[job_id, operation_id] = operation_type(
            start=start_var, end=end_var, interval=intervals, machine=machine_vars)

    # Create and add disjunctive constraints (non-overlap) for each machine.
    for machine_id in range(num_machines):
        model.AddNoOverlap(machine_to_intervals[machine_id])

    # Add precedence constraints between operations in each job (operation i - 1 must finish before operation i starts).
    for job_id, operation_id in zip(job_ids, operation_ids):
        if operation_id == 0:
            continue
        else:
            model.Add(all_operations[job_id, operation_id].start >= all_operations[job_id, operation_id - 1].end)

    # Objective: Minimize makespan (the maximum of the end times of the last operation of each job).
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    last_op_list = []
    unique_job_ids = np.unique(job_ids)
    for job_id in unique_job_ids:
        job_mask = (job_ids == job_id)
        last_op_id = operation_ids[job_mask].max()
        last_op_list.append(all_operations[job_id, last_op_id].end)
    model.AddMaxEquality(obj_var, last_op_list)
    model.Minimize(obj_var)

    # Solve the model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1800.0
    status = solver.Solve(model)

    # Return the results.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        j, o, m, st, end = [], [], [], [], []
        print(f'Makespan: {solver.Value(obj_var)}')
        schedule = []  # Store the schedule details here.
        # for job_id, job in enumerate(data):
        #     for operation_id, operation in enumerate(job):
        for job_id, operation_id in zip(job_ids, operation_ids):
            operation = all_operations[job_id, operation_id]
            machine_assigned = next(
                m_id for m_id, m_var in enumerate(operation.machine) if m_var is not None and solver.Value(m_var))
            start_time = solver.Value(operation.start)
            end_time = solver.Value(operation.end)
            schedule.append((job_id, operation_id, start_time, end_time, machine_assigned))
            print(f'Job {job_id}, Operation {operation_id}, Start {start_time}, End {end_time},'
                  f'Machine {machine_assigned}')
            j.append(job_id)
            o.append(operation_id)
            m.append(machine_assigned)
            st.append(start_time)
            end.append(end_time)
    else:
        print('No feasible solution found.')

    if status == cp_model.OPTIMAL:
        print('Optimal makespan:', solver.ObjectiveValue())
        return [0, solver.ObjectiveValue(), [j, o, m, st, end]]
    elif status == cp_model.FEASIBLE:
        print('Feasible makespan:', solver.ObjectiveValue())
        return [1, solver.ObjectiveValue(), [j, o, m, st, end]]
    else:
        print('Not found any Sol. Return [-1, -1]')
        return [-1, -1, -1]


if __name__ == '__main__':
    env_params = {
        'num_jobs': 20,
        'num_operations': None,
        'num_machines': 5,
        'min_processing_time': 1,
        'max_processing_time': 20,
        'min_eligible_ma_per_op': 1,
        'max_eligible_ma_per_op': None,
    }

    # gen = JobSchedulingProblemGenerator(**env_params)
    gen = FlexibleJobShopProblemGenerator(**env_params)
    # gen = SD1FJSPGenerator(env_params['num_jobs'], env_params['num_machines'])
    job, operation, duration, _, _ = gen.generate_instances(1000)
    result_list = []
    for job_idx, operation_idx, duration in zip(job, operation, duration):
        # s_type, makespan, result = jssp_sat(job_idx, operation_idx, duration)
        s_type, makespan, result = fjsp_sat(job_idx, operation_idx, duration)
        if s_type != -1:
            result_list.append(makespan)
            print("Last 5 makespan:", result_list[-5:])
            print("Average makespan:", np.mean(result_list))
            print("Processing instance:", len(result_list))
    print("Average makespan:", np.mean(result_list))
