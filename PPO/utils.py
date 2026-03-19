import sys
import os
from datetime import datetime
import logging
import logging.config
import shutil
import time
import torch
import numpy as np
import random
from pathlib import Path

from configs.jssp import load_data_from_L2D, load_benchmark_solution_for_JSSP
from configs.fjsp import load_data_from_SD, load_benchmark_solution, get_benchmark_name, load_or_solution_for_FJSP
from configs.ffsp import load_data_from_Matnet_files


def create_logger(env_type, ckpt_path=None, train_flag=True, **params):
    if (ckpt_path is not None) and train_flag:
        folder_path = ckpt_path + '_resume/'
    else:
        id = random.randint(0, 1000)
        id = str(id).zfill(4)
        folder_path = (params['folder_path'] + '/' + env_type +
                       '/' + datetime.now().strftime("%Y%m%d_%H%M%S") + '_'
                       + params['run_name'] + '_'
                       + str(id) + '/')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if 'file_name' in params:
        filename = folder_path + params['file_name'] + '.txt'
    else:
        filename = folder_path + 'log.txt'

    file_mode = 'a' if os.path.isfile(filename) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    return folder_path


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    main_file_copied = False

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if src_abspath == os.path.abspath(sys.argv[0]) and main_file_copied:
                continue
            elif src_abspath == os.path.abspath(sys.argv[0]):
                main_file_copied = True

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


def print_config(logger=lambda x: print(x), **kwargs):
    env_params = kwargs.get('env_params', {})
    model_params = kwargs.get('model_params', {})
    optimizer_params = kwargs.get('optimizer_params', {})
    runner_params = kwargs.get('runner_params', {})
    logger_params = kwargs.get('logger_params', {})
    """
    Helper function to print all parameters before the experiment starts.
    """
    logger("=========================== Logging Experiment Parameters ============================")
    logger("Environment Params: {}".format(env_params))
    logger("Model Params: {}".format(model_params))
    logger("Optimizer Params: {}".format(optimizer_params))
    logger("Runner Params: {}".format(runner_params))
    logger(
        "Logger Params: {}".format({key: val for key, val in logger_params.items() if key != 'changes'}))
    logger("=====================================================================================")


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.start_time_last = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.start_time_last = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count

        remain_time = (curr_time - self.start_time_last) * remain

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        self.start_time_last = curr_time

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


class AverageMeter:
    def __init__(self):
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(data_path):
    if 'L2D' in data_path:
        return load_data_from_L2D(data_path)
    elif 'TNNLS' in data_path:
        return load_data_from_SD(data_path)
    elif 'Matnet' in data_path:
        return load_data_from_Matnet_files(data_path)


# Load Baseline solution for Scheduling
def load_baseline(path):
    # 1.FJSP Benchmark
    if ("BenchData" in path) and ("FJSP" in path):
        bench = get_benchmark_name(path)
        return load_benchmark_solution(bench) if bench else None
    # 2.JSSP Benchmark
    elif ("BenchData" in path) and ("JSSP" in path):
        data_name = os.path.splitext(os.path.basename(path))[0]
        return load_benchmark_solution_for_JSSP(data_name)
    else:
        # 3.No Baseline for FJSP Validation
        if "data_train_vali" in path:
            return None
        else:
            # 4.FJSP Synthetic Data's OR solution as baseline
            d_path = Path(path)
            source, data_name = d_path.parent.name, d_path.name
            file_path = f'../data/FJSP/TNNLS/or_solution/{source}/solution_{data_name}.npy'
            return load_or_solution_for_FJSP(file_path)

if __name__ == '__main__':
    test_dataset_path = [
        # JSSP L2D Benchmark
        '../data/JSSP/L2D/BenchDataNmpy/dmu20x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu20x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu30x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu30x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu40x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu40x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu50x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/dmu50x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai15x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai20x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai20x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai30x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai30x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai50x15.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai50x20.npy',
        '../data/JSSP/L2D/BenchDataNmpy/tai100x20.npy',

        # JSSP L2D dataset
        '../data/JSSP/L2D/Vali/generatedData10_10_Seed200.npy',

        # FJSP Benchmark
        '../data/FJSP/TNNLS/BenchData/Brandimarte',
        '../data/FJSP/TNNLS/BenchData/Hurink_rdata',
        '../data/FJSP/TNNLS/BenchData/Hurink_edata',
        '../data/FJSP/TNNLS/BenchData/Hurink_vdata',

        # Songwen's dataset
        '../data/FJSP/TNNLS/SD1/10x5',
        '../data/FJSP/TNNLS/SD1/15x10',
        '../data/FJSP/TNNLS/SD1/20x5',
        '../data/FJSP/TNNLS/SD1/20x10',
        '../data/FJSP/TNNLS/SD1/30x10',
        '../data/FJSP/TNNLS/SD1/40x10',

        # TNNLS dataset
        '../data/FJSP/TNNLS/SD2/10x5+mix',
        '../data/FJSP/TNNLS/SD2/15x10+mix',
        '../data/FJSP/TNNLS/SD2/20x5+mix',
        '../data/FJSP/TNNLS/SD2/20x10+mix',
        '../data/FJSP/TNNLS/SD2/30x10+mix',
        '../data/FJSP/TNNLS/SD2/40x10+mix',

        '../data/FJSP/TNNLS/data_train_vali/SD1/10x5',

        # Matnet dataset
        '../data/FFSP/Matnet/unrelated_10000_problems_444_job20_2_10.pt',
        '../data/FFSP/Matnet/unrelated_10000_problems_444_job50_2_10.pt',
        '../data/FFSP/Matnet/unrelated_10000_problems_444_job100_2_10.pt',
    ]
    for data_path in test_dataset_path:
        data_temp = load_dataset(data_path)
        baseline_temp = load_baseline(data_path)
        print(f"Dataset loaded from {data_path} with {sum(data_temp[0])} instances.")
        if baseline_temp is not None:
            print(f"Baseline loaded from {data_path} with {len(baseline_temp)} instances.")
        else:
            print(f"No baseline loaded from {data_path}.")
