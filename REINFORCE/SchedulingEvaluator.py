import numpy as np
import torch

from utils import AverageMeter
from SchedulingModel import build_model_input
# from SchedulingModel_old import build_model_input


def validate_model(env, model, dataset, batch_size, inference_type='greedy', sampling_times=100, baseline=None):
    with torch.no_grad():
        assert inference_type in ['greedy', 'sampling', 'aug_sample'], f"Unknown inference_type: {inference_type}"
        model.eval()
        model.set_decode_type('greedy' if inference_type == 'greedy' else 'sampling')

        score_am = AverageMeter()
        score_list = [] if baseline is not None else None

        def run_inference(bs, env, job_batch, op_batch, dur_batch, dependency_batch, connection_batch):
            env.load_data(bs, job_batch, op_batch, dur_batch, dependency_batch, connection_batch)
            state = env.reset_state()
            done = False
            while not done:
                state, _ = build_model_input(state, env)
                action, _ = model(state)
                state, _, done = env.step(action)
            return state.finish_time.max(axis=1).max(axis=1)

        def prepare_batch(data, start, end, repeat=None):
            if repeat is not None:
                return [np.repeat(arr[start:end], repeat, axis=0) for arr in data]
            else:
                return [arr[start:end] for arr in data]

        for instance_num, job_idx, operation_idx, duration, dependency, connection in zip(*dataset):
            data = [job_idx, operation_idx, duration, dependency, connection]
            episode = 0
            while episode < instance_num:
                remaining = instance_num - episode
                bs = min(batch_size, remaining)
                job_batch, op_batch, dur_batch, dependency_batch, connection_batch = prepare_batch(
                    data, episode, episode + bs, repeat=sampling_times if inference_type == 'aug_sample' else None)
                score = run_inference((bs * sampling_times) if inference_type == 'aug_sample' else bs,
                                                 env, job_batch, op_batch, dur_batch, dependency_batch, connection_batch)
                if inference_type == 'aug_sample':
                    score = score.reshape(bs, sampling_times)
                    score = score.min(axis=1)

                score_am.update(score.mean().item(), bs)
                episode += bs
                if baseline is not None:
                    score_list.append(score)

        if baseline is not None:
            # calculate gap
            score_lists = np.concatenate(score_list, axis=0)
            makespan = np.array(baseline)
            gap = (score_lists / makespan - 1) * 100
            gap = np.mean(gap)
        else:
            gap = None

    return score_am.avg, gap