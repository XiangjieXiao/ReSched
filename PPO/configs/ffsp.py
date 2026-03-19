import numpy as np
import torch

# for FFSP in MatNet
env_params = {
    'env_type': 'FFSP',
    'generate_param': {
        'num_jobs': 20,
        'num_machines': 12,
        'num_stages': 3,
        'min_processing_time': 2,
        'max_processing_time': 9,
        'resource_flexibility': True,
        'machine_cnt_list': [4, 4, 4],
        'machine_flexibility': False,
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
    'seed': 42,

    'test_only': True,

    # if ckpt is provided, model_pth will be ignored
    'checkpoint': None,
    'model_path': None,

    'training': {
        'epochs': 2000*5,
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
        'dataset_path': None,
        'gen_instance_num': 100,
    },

    'test': {
        'inference_type': [
            'greedy',
            'sampling',
            'aug_sample',
        ],
        'batch_size': 500,
        'aug_batch_size': 20,
        'sample_times': 24,
        'dataset_path': [
            '../data/FFSP/Matnet/unrelated_10000_problems_444_job20_2_10.pt',
            '../data/FFSP/Matnet/unrelated_10000_problems_444_job50_2_10.pt',
            '../data/FFSP/Matnet/unrelated_10000_problems_444_job100_2_10.pt',
        ],
    },
}

model_params = {
    'action_space': 2,
    'embedding_dim': 128,
    'block_num': 2,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
}

logger_params = {
    'folder_path': '../result',
    'run_name': 'Matnet-20',
    'save_file': True,
}

# Matnet use the first 100 instances for testing.
def load_data_from_Matnet_files(data_path, instance_num=100):
    data = torch.load(data_path)
    duration_list = data['problems_INT_list']
    for i in range(len(duration_list)):
        duration_list[i] = duration_list[i].numpy()[:instance_num]
    instance_size = len(duration_list[0])
    num_job = duration_list[0].shape[1]
    num_stage = len(duration_list)
    num_machine = sum([duration_list[i].shape[2] for i in range(num_stage)])
    machine_cnt_list = [duration_list[i].shape[2] for i in range(num_stage)]

    # [num_job*num_stage]
    job_idx = np.repeat(np.arange(num_job), num_stage)
    stage_idx = np.tile(np.arange(num_stage), num_job)

    # [instance_size, num_job*num_stage, num_machine]
    duration = np.zeros((instance_size, num_job * num_stage, num_machine), dtype=int)
    for stage_id in range(num_stage):
        machine_idx = np.arange(machine_cnt_list[stage_id]) + (sum(machine_cnt_list[:stage_id]) if stage_id > 0 else 0)
        d = duration[:, stage_idx == stage_id, :]
        d[:, :, machine_idx] = duration_list[stage_id]
        duration[:, stage_idx == stage_id, :] = d

    dependency = np.zeros((num_job * num_stage, num_job * num_stage), dtype=bool)
    connection = np.zeros((num_job * num_stage, num_job * num_stage), dtype=bool)

    for i in range(num_job * num_stage):
        # each job's first stage has no predecessor
        if stage_idx[i] != 0:
            # each stage's predecessor is the previous stage of the same job
            dep_from = i - 1
            dep_to = i
            dependency[dep_from, dep_to] = True

            num2end = num_stage - stage_idx[i]
            connection[dep_from, dep_to:(dep_to + num2end)] = True

    job_idx = job_idx[None, :].repeat(instance_size, axis=0)
    stage_idx = stage_idx[None, :].repeat(instance_size, axis=0)
    dependency = dependency[None, :, :].repeat(instance_size, axis=0)
    connection = connection[None, :, :].repeat(instance_size, axis=0)

    return [instance_size], [job_idx], [stage_idx], [duration], [dependency], [connection]
