import re
import os
import numpy as np
from collections import defaultdict
import pandas as pd


# for FJSP
env_params = {
    'env_type': 'FJSP',
    'generate_param': {
        'num_jobs':  10,
        'num_machines': 5,
        'max_processing_time': 20,

        # only for FJSP
        'SD1': True,        # some parameter will be automatically set in SD1 setting
        # 'SD1': False,
    },
}

optimizer_params = {
    'optimizer': {
        'lr': 6e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [500*5, 1000*5, 1500*5],
        'gamma': 0.5,
    }
}

runner_params = {
    'use_cuda': True,
    'cuda_device_num': 0,
    'seed': 114514,

    'test_only': True,

    # if ckpt is provided, model_pth will be ignored
    'checkpoint': None,
    # 'model_path': None,
    'model_path': '../ckpt/PPO/SD1-10x05.pth',

    'training': {
        'epochs': 2000*5,     # Large training budget
        # # 'epochs': 400,       # Small training budget
        'episode': 50,
        'sample_batch_size': 50,
        'mini_batch_size': 50,
        'K_epochs': 4,

        'clip_range': 0.2,
        'vf_coef': 0.5,
        'entropy_coef': 0.01,
        # discount factor
        'gamma': 1,
        'lambda': 0.98,
    },
    'validation': {
        'batch_size': 200,
        'dataset_path': '../data/FJSP/TNNLS/data_train_vali/SD1/10x5',
        'gen_instance_num': None,
    },

    'test': {
        'inference_type': [
            'greedy',
            # 'sampling',
            # 'aug_sample',
        ],

        # for FJSP in TNNLS
        'batch_size': 200,
        'aug_batch_size': 1,
        'sample_times': 100,
        'dataset_path': [
            # FJSP Benchmark
            '../data/FJSP/TNNLS/BenchData/Brandimarte',
            '../data/FJSP/TNNLS/BenchData/Hurink_rdata',
            '../data/FJSP/TNNLS/BenchData/Hurink_edata',
            '../data/FJSP/TNNLS/BenchData/Hurink_vdata',

            # Songwen's dataset
            '../data/FJSP/TNNLS/SD1/10x5',
            # '../data/FJSP/TNNLS/SD1/20x5',
            # '../data/FJSP/TNNLS/SD1/15x10',
            # '../data/FJSP/TNNLS/SD1/20x10',
            # '../data/FJSP/TNNLS/SD1/30x10',
            # '../data/FJSP/TNNLS/SD1/40x10',

            # TNNLS dataset
            '../data/FJSP/TNNLS/SD2/10x5+mix',
            # '../data/FJSP/TNNLS/SD2/20x5+mix',
            # '../data/FJSP/TNNLS/SD2/15x10+mix',
            # '../data/FJSP/TNNLS/SD2/20x10+mix',
            # '../data/FJSP/TNNLS/SD2/30x10+mix',
            # '../data/FJSP/TNNLS/SD2/40x10+mix',
        ],
    },
}

model_params = {
    'action_space': 2,  # 0: processing-time ！= 0; 1: processing-time == 0
    'embedding_dim': 128,
    'block_num': 2,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
}

logger_params = {
    'folder_path': '../result',
    'run_name': 'SD1-10x05-PPO',
    'save_file': False,
}

######################################################################
# Load data from SD files
######################################################################
def load_data_from_SD(data_path):
    data1, data2 = load_data_from_SD_files(data_path)

    temp_job_idx, temp_operation_idx, temp_duration, temp_dependency, temp_connection = [], [], [], [], []
    for num_op_per_job, pt in zip(data1, data2):
        job_id = np.concatenate([np.full(value, idx) for idx, value in enumerate(num_op_per_job)])
        operation_id = np.concatenate([np.arange(value) for value in num_op_per_job])

        ################################################################################################################
        # generate dependency matrix(o2o, j2o)
        # [from, to]
        dep_on = np.zeros((sum(num_op_per_job), sum(num_op_per_job)), dtype=np.bool_)
        dep_oj = np.zeros((sum(num_op_per_job), sum(num_op_per_job)), dtype=np.bool_)
        job_count = -1
        for idx in range(len(operation_id)):
            # each job's first operation has no predecessor
            if operation_id[idx] != 0:
                # each operation's predecessor is the previous operation of the same job
                dep_from = idx - 1
                dep_to = idx
                dep_on[dep_from, dep_to] = True

                num2end = num_op_per_job[job_count] - operation_id[idx]
                dep_oj[dep_from, dep_to:(dep_to + num2end)] = True
            elif operation_id[idx] == 0:
                job_count += 1

        temp_job_idx.append(job_id)
        temp_operation_idx.append(operation_id)
        temp_duration.append(pt)
        temp_dependency.append(dep_on)
        temp_connection.append(dep_oj)

    # group by duration(only duration have operation and machine information)
    group_idx = assign_group_ids(temp_duration)
    job_idx, operation_idx, duration, dependency_on, dependency_oj = \
        group_and_batch_by_idx(group_idx, temp_job_idx, temp_operation_idx, temp_duration, temp_dependency,
                               temp_connection)

    instance_size = [len(job) for job in job_idx]

    return instance_size, job_idx, operation_idx, duration, dependency_on, dependency_oj


def load_data_from_SD_files(directory):
    """
        load all files within the specified directory
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        return [], []

    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
        for f in files:
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return dataset_job_length, dataset_op_pt


def text_to_matrix(text):
    """
            Convert text form of the data into matrix form
    :param text: the standard text form of the instance
    :return:  the matrix form of the instance
            job_length: the number of operations in each job (shape [J])
            op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    """
    n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
    n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

    job_length = np.zeros(n_j, dtype='int32')
    op_pt = []

    for i in range(n_j):
        content = np.array([int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
        job_length[i] = content[0]

        idx = 1
        for j in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype='int32')
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt

            idx = next_idx
            op_pt.append(op_pt_row)

    op_pt = np.array(op_pt)

    return job_length, op_pt


def assign_group_ids(array_list):
    shape_to_id = {}
    group_ids = []

    current_id = 0
    for arr in array_list:
        shape = arr.shape
        if shape not in shape_to_id:
            shape_to_id[shape] = current_id
            current_id += 1
        group_ids.append(shape_to_id[shape])

    return group_ids


def group_and_batch_by_idx(group_idx, *lists):
    grouped = [defaultdict(list) for _ in lists]

    for i, gid in enumerate(group_idx):
        for var_list, grouped_dict in zip(lists, grouped):
            grouped_dict[gid].append(var_list[i])

    batched_result = []
    for g in grouped:
        batch_group = []
        for gid in sorted(g.keys()):
            batch_group.append(np.stack(g[gid], axis=0))
        batched_result.append(batch_group)

    return batched_result


def load_benchmark_solution(data_name):
    """
        load the best solutions of benchmark data from files
    :param data_name: the name of benchmark data
    :return: makespan, average makespan
    """
    file_path = f'../data/FJSP/TNNLS/BenchData/BenchDataSolution.csv'
    bench_data = pd.read_csv(file_path)
    make_span = bench_data.loc[bench_data['benchname'] == data_name, 'ub'].values
    return make_span


def get_benchmark_name(d_path):
    # for FJSP benchmark
    if "Brandimarte" in d_path:
        return "Brandimarte"
    elif "Hurink_rdata" in d_path:
        return "Hurink_rdata"
    elif "Hurink_edata" in d_path:
        return "Hurink_edata"
    elif "Hurink_vdata" in d_path:
        return "Hurink_vdata"
    else:
        return None


def load_or_solution_for_FJSP(file_path):
    """
        load the results solved by OR-Tools
    """
    if os.path.exists(file_path):
        solution = np.load(file_path)
        or_make_span = solution[:, 0]
        return or_make_span
    else:
        return None
