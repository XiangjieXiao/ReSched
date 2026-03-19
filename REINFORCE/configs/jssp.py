import numpy as np

# Configuration for JSSP (Job Shop Scheduling Problem) in L2D (Learning to Dispatch)
# for JSSP in L2D
env_params = {
    'env_type': 'JSSP',
    'generate_param': {
        'num_jobs':  10,
        'num_machines': 10,
        'min_processing_time': 1,
        'max_processing_time': 99,
    },
}

optimizer_params = {
    'optimizer': {
        'lr': 5e-5,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [500, 1000, 1500],
        'gamma': 0.5,
    }
}

runner_params = {
    'use_cuda': True,
    'cuda_device_num': 0,
    'seed': 42,

    'test_only': True,

    # if ckpt is provided, model_pth will be ignored
    'checkpoint': None,
    # 'model_path': None,
    'model_path': '../ckpt/REINFORCE/JSSP/Old-JSSP-10x10.pth',

    'training': {
        'epochs': 2000,
        'episode': 1000,
        'batch_size': 50,
        'discount_factor': 0.99,
    },
    'validation': {
        'batch_size': 200,
        'dataset_path': '../data/JSSP/L2D/Vali/generatedData10_10_Seed200.npy',
        'gen_instance_num': None,
    },

    'test': {
        'inference_type': [
            'greedy',
            # 'sampling',
            # 'aug_sample',
        ],
        'batch_size': 200,
        'aug_batch_size': 2,
        'sample_times': 128,
        'dataset_path': [
            # JSSP L2D dataset
            '../data/JSSP/L2D/generatedData6_6_Seed200.npy',
            # '../data/JSSP/L2D/generatedData10_10_Seed200.npy',
            # '../data/JSSP/L2D/generatedData15_15_Seed200.npy',
            # '../data/JSSP/L2D/generatedData20_20_Seed200.npy',
            # '../data/JSSP/L2D/generatedData30_20_Seed200.npy',
            # '../data/JSSP/L2D/generatedData50_20_Seed200.npy',
            # # '../data/JSSP/L2D/generatedData100_20_Seed200.npy',

            # JSSP L2D Benchmark(Tai)
            '../data/JSSP/L2D/BenchDataNmpy/tai15x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/tai20x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/tai20x20.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/tai30x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/tai30x20.npy',
            #
            # '../data/JSSP/L2D/BenchDataNmpy/tai50x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/tai50x20.npy',
            # # '../data/JSSP/L2D/BenchDataNmpy/tai100x20.npy',

            # JSSP L2D Benchmark(DMU)
            '../data/JSSP/L2D/BenchDataNmpy/dmu20x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/dmu20x20.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/dmu30x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/dmu30x20.npy',
            #
            # '../data/JSSP/L2D/BenchDataNmpy/dmu40x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/dmu40x20.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/dmu50x15.npy',
            # '../data/JSSP/L2D/BenchDataNmpy/dmu50x20.npy',
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
    'run_name': 'JSSP-10x10',
    'save_file': True,
}

def load_data_from_L2D(data_path):
    ori_data = np.load(data_path)
    instance_size = ori_data.shape[0]
    num_job = ori_data.shape[2]
    num_operation = ori_data.shape[3]
    # num_machine is equal to num_operation in L2D
    num_machine = num_operation

    # [instance_size, num_job*num_operation]
    duration_value = ori_data[:, 0, :, :].reshape(instance_size, -1)
    machine_connection = ori_data[:, 1, :, :].reshape(instance_size, -1) - 1
    duration = np.zeros((instance_size, num_job * num_operation, num_machine), dtype=np.float32)
    duration[np.arange(instance_size)[:, None], np.arange(num_job * num_operation)[None,
                                                :], machine_connection] = duration_value

    # [instance_size, num_job*num_operation]
    job_id = np.repeat(np.arange(num_job), num_operation)
    operation_id = np.tile(np.arange(num_operation), num_job)

    dep_on = np.zeros((num_job * num_operation, num_job * num_operation), dtype=bool)
    dep_oj = np.zeros((num_job * num_operation, num_job * num_operation), dtype=bool)

    for idx in range(num_job * num_operation):
        # each job's first operation has no predecessor
        if operation_id[idx] != 0:
            # each operation's predecessor is the previous operation of the same job
            dep_from = idx - 1
            dep_to = idx
            dep_on[dep_from, dep_to] = True

            num2end = num_operation - operation_id[idx]
            dep_oj[dep_from, dep_to:(dep_to + num2end)] = True

    job_idx = job_id[None, :].repeat(instance_size, axis=0)
    operation_idx = operation_id[None, :].repeat(instance_size, axis=0)
    dependency_on = dep_on[None, :, :].repeat(instance_size, axis=0)
    dependency_oj = dep_oj[None, :, :].repeat(instance_size, axis=0)

    return [instance_size], [job_idx], [operation_idx], [duration], [dependency_on], [dependency_oj]



def load_benchmark_solution_for_JSSP(data_name: str):
    """
    Load best-known makespan (UB) for JSSP benchmarks.
    Supported keys: 'tai15x15', ..., 'dmu50x20'.
    Returns a (10,) np.int64 array or None if key not found.
    """
    _BENCHMARK_MAKESPANS = {
        'tai15x15': [1231, 1244, 1218, 1175, 1224, 1238, 1227, 1217, 1274, 1241],
        'tai20x15': [1357, 1367, 1342, 1345, 1339, 1360, 1462, 1396, 1332, 1348],
        'tai20x20': [1642, 1600, 1557, 1644, 1595, 1645, 1680, 1603, 1625, 1584],
        'tai30x15': [1764, 1784, 1791, 1828, 2007, 1819, 1771, 1673, 1795, 1670],
        'tai30x20': [2006, 1939, 1846, 1979, 2000, 2006, 1889, 1937, 1960, 1923],
        'tai50x15': [2760, 2756, 2717, 2839, 2679, 2781, 2943, 2885, 2655, 2723],
        'tai50x20': [2868, 2869, 2755, 2702, 2725, 2845, 2825, 2784, 3071, 2995],
        'tai100x20': [5464, 5181, 5568, 5339, 5392, 5342, 5436, 5394, 5358, 5183],

        # 20x15: Dmu01–05 (rcmax) + Dmu41–45 (cscmax)
        'dmu20x15': [2563, 2706, 2731, 2669, 2749, 3248, 3390, 3441, 3475, 3266],
        # 20x20: Dmu06–10 (rcmax) + Dmu46–50 (cscmax)
        'dmu20x20': [3244, 3046, 3188, 3092, 2984, 4035, 3939, 3763, 3706, 3729],
        # 30x15: Dmu11–15 (rcmax) + Dmu51–55 (cscmax)
        'dmu30x15': [3430, 3492, 3681, 3394, 3343, 4156, 4297, 4378, 4361, 4258],
        # 30x20: Dmu16–20 (rcmax) + Dmu56–60 (cscmax)
        'dmu30x20': [3750, 3812, 3844, 3764, 3699, 4939, 4647, 4701, 4607, 4721],
        # 40x15: Dmu21–25 (rcmax) + Dmu61–65 (cscmax)
        'dmu40x15': [4380, 4725, 4668, 4648, 4164, 5169, 5247, 5312, 5226, 5173],
        # 40x20: Dmu26–30 (rcmax) + Dmu66–70 (cscmax)
        'dmu40x20': [4647, 4848, 4692, 4691, 4732, 5701, 5779, 5763, 5688, 5868],
        # 50x15: Dmu31–35 (rcmax) + Dmu71–75 (cscmax)
        'dmu50x15': [5640, 5927, 5728, 5385, 5635, 6207, 6463, 6136, 6196, 6189],
        # 50x20: Dmu36–40 (rcmax) + Dmu76–80 (cscmax)
        'dmu50x20': [5621, 5851, 5713, 5747, 5577, 6718, 6747, 6755, 6910, 6634],
    }
    arr = _BENCHMARK_MAKESPANS.get(data_name)
    if arr is None:
        return None
    return np.array(arr, dtype=np.int64)
