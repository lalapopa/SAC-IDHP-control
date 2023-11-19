import copy
import json
import os
import numpy as np
from tqdm import tqdm

from envs.citation import Citation
from envs.citation_attitude import CitationAttitude
from tasks.tracking_test import TrackAltitudeEdit
from tools import set_plot_styles, nMAE

from agents import SAC
from tools.utils import d2r
from tools import set_random_seed

CONFIG_ENV_CITATION = {
    "seed": None,
    "h0": 2000,  # initial trimmed altitude
    "v0": 90,  # initial trimmed airspeed
    "trimmed": False,  # trimmed initial action is known
    "failure": None,  # failure type
    "failure_time": 10,  # failure time [s]
    "sensor_noise": False,
}
CONFIG_TASK_ALTITUDE = {
    "T": 20,  # task duration
    "dt": 0.01,  # time-step
    "tracking_scale": {  # tracking scaling factors
        "h": 1 / 240,
        "theta": 1 / d2r(30),
        "phi": 1 / d2r(30),
        "beta": 1 / d2r(7),
    },
    "tracking_thresh": {  # tracking threshold on rmse used for adaptive lr
        "h": None,
        "theta": None,
        "phi": None,
        "beta": None,
    },
}


def main():
    save_dir = "trained/SAC_citation_attitude_tracking_altitude_1659276263/764000"
    config_env = copy.deepcopy(CONFIG_ENV_CITATION)
    config_task = copy.deepcopy(CONFIG_TASK_ALTITUDE)
    configfile = open(os.path.join(save_dir, "config.json"), "r")
    config_agent = json.load(configfile)
    inner_save_dir = config_agent["agent_inner"]

    task = TrackAltitudeEdit(config_task, evaluate=True)
    env = Citation(config_env, dt=config_task["dt"], obs_extra=[9])

    obs = env.reset()
    action_env = env.action
    print(f"{action_env =}")

    for t in tqdm(range(2000)):  # TODO: Play with this function
        # Get action
        action_env = np.array([0, 0, 0])

        # Take action
        obs, _ = env.step(action_env)

        # Check for crash
        if np.isnan(obs).sum() > 0:
            break

        # Sample tracking reference and error

    env.render(task)
    input()


if __name__ == "__main__":
    main()
