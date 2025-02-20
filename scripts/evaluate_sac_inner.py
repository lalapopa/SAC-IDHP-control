import copy
import csv
import json
import sys
import os
from termcolor import colored
import numpy as np

from tools import set_plot_styles, nMAE
from tasks import TrackAttitude
from envs import Citation

from agents import SAC
from tools import set_random_seed
from tensorflow.python.ops.numpy_ops import np_config

from tools.utils import d2r

set_plot_styles()
os.system("color")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np_config.enable_numpy_behavior()

# Config
CONFIG_ENV_CITATION = {
    "seed": None,
    "h0": 2000,  # initial trimmed altitude
    "v0": 90,  # initial trimmed airspeed
    "trimmed": False,  # trimmed initial action is known
    "failure": None,  # failure type
    "failure_time": 10,  # failure time [s]
    "sensor_noise": False,
}
CONFIG_TASK_ATTITUDE = {
    "T": 30,  # task duration
    "dt": 0.01,  # time-step
    "tracking_scale": {  # tracking scaling factors
        "theta": 1 / d2r(30),
        "phi": 1 / d2r(30),
        "beta": 1 / d2r(7),
    },
    "tracking_thresh": {  # tracking threshold on rmse used for adaptive lr
        "theta": None,
        "phi": None,
        "beta": None,
    },
}


def main(save_dir=None):
    # Evaluate
    if save_dir:
        evaluate(save_dir, find_best=True, find_best_extensive=True)
        return

    # Trained agent:
    evaluate("trained/SAC_citation_tracking_attitude_1659223622/496000")
    input()


def evaluate(save_dir, find_best=False, find_best_extensive=False):
    """
    Evaluate the SAC agent
    """

    # Config
    configfile = open(os.path.join(save_dir, "config.json"), "r")
    config_agent = json.load(configfile)
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ATTITUDE)

    # Randomize
    seed = config_agent["seed"]
    config_env["seed"] = seed
    set_random_seed(seed)

    # Task
    task = TrackAttitude(config_task, evaluate_hard=True)
    # task = TrackAttitude(config_task, evaluate=True)
    # task = TrackAttitude(config_task, train_online=True)

    # Environment
    # config_env["failure"] = "dr_stuck"
    # config_env["failure_time"] = 30
    env = Citation(config_env, dt=config_task["dt"])

    # Find best performing agent on task
    if find_best:
        print(save_dir)

        # CSV log
        fieldnames = ["agent", "nmae", "loss_t"]
        logfile = open(
            os.path.join("trained_evaluated", os.path.basename(os.path.dirname(save_dir)) + "_" + str(task) + ".csv"),
            "wt+",
        )
        logger = csv.DictWriter(logfile, fieldnames=fieldnames)
        logger.writeheader()
        logfile.flush()

        # Get all best saved agents
        save_dirs_all = [_.path for _ in os.scandir(os.path.dirname(save_dir)) if _.is_dir() and _.name.isnumeric()]
        save_dirs_best = [
            os.path.join(os.path.dirname(os.path.dirname(_)), _.name)
            for _ in os.scandir(save_dir)
            if _.is_file() and _.name.isnumeric()
        ]

        # Overwrite
        save_dirs_all = [
            "trained/SAC_citation_tracking_attitude_1659223622/522000",
            "trained/SAC_citation_tracking_attitude_1659223622/898000",
            "trained/SAC_citation_tracking_attitude_1659223622/496000",
            "trained/SAC_citation_tracking_attitude_1659223622/488000",
            "trained/SAC_citation_tracking_attitude_1659223622/994000",
            "trained/SAC_citation_tracking_attitude_1659223622/992000",
            "trained/SAC_citation_tracking_attitude_1659223622/896000",
            "trained/SAC_citation_tracking_attitude_1659223622/996000",
            "trained/SAC_citation_tracking_attitude_1659223622/592000",
            "trained/SAC_citation_tracking_attitude_1659223622/998000",
            "trained/SAC_citation_tracking_attitude_1659223622/610000",
            "trained/SAC_citation_tracking_attitude_1659223622/920000",
            "trained/SAC_citation_tracking_attitude_1659223622/990000",
        ]

        save_dirs_all.sort(key=lambda x: int(os.path.basename(x)), reverse=True)
        save_dirs_best.sort(key=lambda x: int(os.path.basename(x)), reverse=True)

        if find_best_extensive:
            save_dirs = save_dirs_all
            # save_dirs = np.array(save_dirs)[save_dirs.index(save_dirs_best[0]) :]
        else:
            save_dirs = save_dirs_best

        for i, save_dir in enumerate(save_dirs):
            # Get new environment
            env = Citation(config_env, dt=config_task["dt"])

            # Load agent
            agent = SAC.load(save_dir, task, env)

            # Evaluate
            episode_return = agent.evaluate(crash_attitude=True)

            print(f"agent {os.path.basename(save_dir)}\t[{i+1}/{len(save_dirs)}]:\t\t", end="")

            if episode_return:
                # Metrics: NMAE
                nmae = nMAE(agent, env)
                color = "green" if nmae <= 0.05 else "red"
                print(f"nMAE: {colored(f'{nmae * 100 :.2f}', color)}%\t", end="")

                # Metrics: Temporal loss
                action_loss_t = 0
                action_history = np.array(env.action_history)
                for j in range(1, action_history.shape[0]):
                    action_loss_t += np.sum((action_history[j] - action_history[j - 1]) ** 2)
                print(f"loss_t: {action_loss_t :.3f}")

                # Log
                logger.writerow(
                    {
                        "agent": os.path.basename(save_dir),
                        "nmae": nmae,
                        "loss_t": action_loss_t,
                    }
                )
                logfile.flush()

            else:
                print("crash")
    else:
        # Load agent
        agent = SAC.load(save_dir, task, env)

        # Evaluate
        agent.evaluate()

        # Plots
        env.render(task, show_rmse=False)

        # Metrics: NMAE
        nmae = nMAE(agent, env)
        color = "green" if nmae <= 0.05 else "red"
        print(f"nMAE: {colored(f'{nmae * 100 :.2f}', color)}%\t", end="")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()

    elif len(sys.argv) == 2:
        save_dir = sys.argv[1]
        main(save_dir=save_dir)

    else:
        raise Exception("Incorrect number of arguments provided.")
