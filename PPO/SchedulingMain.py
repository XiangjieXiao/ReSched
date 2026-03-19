from SchedulingRunner import Runner
import importlib

PROBLEM = 'fjsp'   # 'fjsp', 'jssp', 'ffsp'


def main():
    cfg = importlib.import_module(f'configs.{PROBLEM}')
    logger_params = cfg.logger_params
    env_params = cfg.env_params
    model_params = cfg.model_params
    optimizer_params = cfg.optimizer_params
    runner_params = cfg.runner_params

    runner = Runner(
        logger_params=logger_params,
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        runner_params=runner_params,
    )
    runner.run()


if __name__ == '__main__':
    main()

