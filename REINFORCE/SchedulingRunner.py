from logging import getLogger
from pathlib import Path

from utils import (
    create_logger,
    copy_all_src,
    TimeEstimator,
    print_config,
    load_dataset,
    load_baseline,
)

import torch

from SchedulingEnvironment import (
    JSSPEnv,
    FJSPEnv,
    FFSPEnv,
)
from SchedulingModel import Model
# from SchedulingModel_old import Model
from REINFORCETrainer import Trainer
from SchedulingEvaluator import validate_model


class Runner:
    def __init__(
        self,
        logger_params,
        env_params,
        model_params,
        optimizer_params,
        runner_params,
    ):
        self.logger_params = logger_params
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.runner_params = runner_params

        self.device = self._configure_device()
        torch.set_default_dtype(torch.float32)

        self.model = Model(**model_params)
        self.test_only_flag = self.runner_params.get('test_only', False)
        self.trainer = Trainer(self.model, optimizer_params, self.runner_params['training'])
        self.epoch = 1

        self.logger = getLogger(name='runner')
        self.trainer.set_logger(self.logger.info)
        self.save_dir = create_logger(
            env_params['env_type'],
            self.runner_params['checkpoint'],
            not self.test_only_flag,
            **logger_params,
        )

        self.checkpoint = None
        self.validation_freq = self.runner_params['training'].get('validation_freq', 10)
        self._initialize_checkpoint_and_model()

        self.seed = self.runner_params['seed'] if 'seed' in self.runner_params.keys() else None
        self.env, self.env_val = self.env_init()

        self.dataset = {}
        self.baseline = {}
        self.dataset_init()

        print_config(
            logger=self.logger.info,
            env_params=env_params,
            model_params=model_params,
            optimizer_params=optimizer_params,
            runner_params=runner_params,
            logger_params=logger_params,
        )
        if logger_params['save_file']:
            copy_all_src(self.save_dir)
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        if not self.test_only_flag:
            self.train()
            self._clear_cuda_cache()
            self._load_model_after_training()

        self.logger.info('Start testing...')
        self.run_test()

    def train(self):
        total_epochs = self.runner_params['training']['epochs']
        for self.epoch in range(self.epoch, total_epochs + 1):
            score_t, loss_t, model_param = self.trainer.train(self.env, first_epoch=(self.epoch == 1))
            self._log_training_progress(score_t, loss_t, model_param)

    def run_test(self):
        test_params = self.runner_params['test']
        dataset_paths = test_params['dataset_path']

        for inference_type in test_params['inference_type']:
            batch_size = test_params['aug_batch_size'] if inference_type == 'aug_sample' else test_params['batch_size']
            for dataset_path, dataset, baseline in zip(dataset_paths, self.dataset['test'], self.baseline['test']):
                score_test, gap = validate_model(self.env_val, self.model, dataset, batch_size,
                                                 inference_type=inference_type,
                                                 sampling_times=test_params['sample_times'],
                                                 baseline=baseline)
                if gap is not None:
                    self.logger.info('Score: {:.4f} Gap: {:.4f}% on Test dataset path {} with {} inference'.format(
                        score_test, gap, dataset_path, inference_type))
                else:
                    self.logger.info('Score: {:.4f} on Test dataset path {} with {} inference'.format(
                        score_test, dataset_path, inference_type))

    def _configure_device(self):
        use_cuda = self.runner_params['use_cuda']
        if use_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is requested but not available.')
            cuda_device_num = self.runner_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            return torch.device(f'cuda:{cuda_device_num}')
        return torch.device('cpu')

    def _initialize_checkpoint_and_model(self):
        checkpoint_dir = self.runner_params.get('checkpoint')
        model_path = self.runner_params.get('model_path')

        if checkpoint_dir is not None:
            if self.test_only_flag:
                self._load_model_weights(self._best_model_path(checkpoint_dir))
            else:
                self.load_checkpoint()
            return

        if not self.test_only_flag:
            self.checkpoint = self._build_empty_checkpoint()

        if model_path is not None:
            self._load_model_weights(model_path)

    def _build_empty_checkpoint(self):
        return {
            'training': {
                'epochs': [],
                'scores': [],
                'losses': [],
            },
            'validation': {
                'epochs': [],
                'scores': [],
            },
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'scheduler_state_dict': None,
        }

    def _clear_cuda_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _load_model_weights(self, path):
        model_path = Path(path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.logger.info('Model loaded from {}'.format(model_path))

    def _best_model_path(self, directory):
        return Path(directory) / 'best_model.pth'

    def _checkpoint_path(self, directory):
        return Path(directory) / 'checkpoint.pth'

    def _load_model_after_training(self):
        candidate_paths = [self._best_model_path(self.save_dir)]
        checkpoint_dir = self.runner_params.get('checkpoint')
        if checkpoint_dir is not None:
            candidate_paths.append(self._best_model_path(checkpoint_dir))

        # if resume from ckpt but do not get new best model, load the best model from checkpoint dir
        for model_path in candidate_paths:
            if model_path.exists():
                self._load_model_weights(model_path)
                return

        raise FileNotFoundError(
            'No best_model.pth found in {}'.format(', '.join(str(path.parent) for path in candidate_paths))
        )

    def _log_training_progress(self, score_t, loss_t, model_param):
        self.logger.info('=' * 66)
        self.checkpoint['training']['epochs'].append(self.epoch)
        self.checkpoint['training']['scores'].append(score_t)
        self.checkpoint['training']['losses'].append(loss_t)
        self.logger.info('Epoch: {:3d} Score: {:.4f},  Loss: {:.4f}'.format(self.epoch, score_t, loss_t))
        elapsed_time_str, remain_time_str = (
            self.time_estimator.get_est_string(self.epoch, self.runner_params['training']['epochs']))
        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            self.epoch, self.runner_params['training']['epochs'], elapsed_time_str, remain_time_str))

        if (self.epoch % self.validation_freq == 0) or self.epoch == 1:
            self.model.load_state_dict(model_param)
            score_v, gap = validate_model(
                self.env_val,
                self.model,
                self.dataset['validation'],
                batch_size=self.runner_params['validation']['batch_size'],
                baseline=self.baseline['validation'],
            )
            self.checkpoint['validation']['epochs'].append(self.epoch)
            self.checkpoint['validation']['scores'].append(score_v)
            if gap is not None:
                self.logger.info('Epoch: {:3d} Validation Score: {:.4f} Gap: {:.4f}%'.format(
                    self.epoch, score_v, gap))
            else:
                self.logger.info('Epoch: {:3d} Validation Score: {:.4f}'.format(self.epoch, score_v))
            self.save_checkpoint(score_v)

    def env_init(self):
        env_type = self.env_params['env_type']
        sd1_param = self.env_params['generate_param'].get('SD1', None)

        env_classes = {
            'JSSP': lambda param: JSSPEnv(param),
            'FJSP': lambda param: FJSPEnv(param, sd1_param),
            'FFSP': lambda param: FFSPEnv(param),
        }

        if env_type not in env_classes:
            raise ValueError(f"Invalid Env Type: {env_type}")

        env_constructor = env_classes[env_type]

        env = env_constructor(self.env_params['generate_param'])
        env_val = env_constructor(None)

        if self.seed is not None:
            env.set_seed(self.seed)
            env_val.set_seed(self.seed)
            self.logger.info('Data is generated with seed {}.'.format(self.seed))
        else:
            self.logger.info('Data is generated with random.')

        return env, env_val

    def dataset_init(self):
        val_param = self.runner_params['validation']
        test_param = self.runner_params['test']

        # 1. Validation dataset
        if not self.test_only_flag:
            self.logger.info('=' * 66)
            self.logger.info('[Validation] Initializing in-distribution dataset...')
            if val_param['dataset_path'] is not None:
                self.dataset['validation'] = load_dataset(val_param['dataset_path'])
                self.baseline['validation'] = load_baseline(val_param['dataset_path'])
                self.logger.info(f"Validation dataset loaded from {val_param['dataset_path']}")
            else:
                original_tuple = self.env.generator.generate_instances(val_param['gen_instance_num'])
                processed_tuple = ([val_param['gen_instance_num']],) + tuple([[item] for item in original_tuple])
                self.dataset['validation'] = processed_tuple
                self.baseline['validation'] = None
                self.logger.info(f"Validation dataset generated with {val_param['gen_instance_num']} "
                                 f"instances by same generator as training data.")

        # 2. Test dataset
        self.logger.info('=' * 66)
        self.logger.info('[Test] Loading test dataset...')
        self.dataset['test'], self.baseline['test'] = [], []
        for path in test_param['dataset_path']:
            self.dataset['test'].append(load_dataset(path))
            self.baseline['test'].append(load_baseline(path))
            self.logger.info(f'Loaded test dataset from {path}')

    def load_checkpoint(self):
        ckpt_path = self._checkpoint_path(self.runner_params['checkpoint'])
        self.checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.epoch = self.checkpoint['training']['epochs'][-1] + 1

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.logger.info('Model loaded from {}'.format(ckpt_path))

        self.trainer.load_checkpoint(
            self.checkpoint['optimizer_state_dict'],
            self.checkpoint['scheduler_state_dict'],
            ckpt_path,
        )

    def save_checkpoint(self, validation_score):
        self.checkpoint.update(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            scheduler_state_dict=self.trainer.scheduler.state_dict(),
        )
        path_name = self._checkpoint_path(self.save_dir)
        torch.save(self.checkpoint, path_name)
        self.logger.info('Model saved to {}'.format(path_name))

        if validation_score <= min(self.checkpoint['validation']['scores']):
            best_model_path = self._best_model_path(self.save_dir)
            torch.save(self.model.state_dict(), best_model_path)
            self.logger.info('Best model updated(Validation Score: {:.4f})'.format(validation_score))
            self.logger.info('Best model saved to {}'.format(best_model_path))
