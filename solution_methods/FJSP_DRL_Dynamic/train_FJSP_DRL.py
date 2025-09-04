# GITHUB REPO: https://github.com/songwenas12/fjsp-drl

# Code based on the paper:
# "Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning"
# by Wen Song, Xinyang Chen, Qiqiang Li and Zhiguang Cao
# Presented in IEEE Transactions on Industrial Informatics, 2023.
# Paper URL: https://ieeexplore.ieee.org/document/9826438

import os
import sys
import argparse
import logging
import random
import time
import copy
from collections import deque
from pathlib import Path
import numpy as np
import torch
from visdom import Visdom

from solution_methods.helper_functions import load_parameters, initialize_device, set_seeds
from solution_methods.FJSP_DRL_Dynamic.src import PPO as PPO_model
from solution_methods.FJSP_DRL_Dynamic.src.case_generator import CaseGenerator
from solution_methods.FJSP_DRL_Dynamic.src.env_training import FJSPEnv_training
from solution_methods.FJSP_DRL_Dynamic.src.validate import get_validate_env, validate

# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

PARAM_FILE = str(base_path) + "/configs/FJSP_DRL_Dynamic.toml"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def train_FJSP_DRL(**parameters):
    logging.info("Training started.")

    # Retrieve parameters
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["model_parameters"]
    train_parameters = parameters["train_parameters"]
    curriculum_parameters = parameters.get("curriculum_parameters", {"enabled": False})

    # Initialize device and set seeds
    device = initialize_device(parameters)
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default tensor type for device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device.type == 'cuda' else 'torch.FloatTensor')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Prepare model dimensions and validation parameters
    model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
    model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

    # Initialize model and memories
    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_parameters, train_parameters, num_envs=env_parameters["batch_size"])
    
    best_models = deque()
    makespan_best = float("inf")

    # Setup for visualization if enabled
    is_viz = train_parameters["viz"]
    viz = Visdom(env=train_parameters["viz_name"]) if is_viz else None

    # Generate directories for saving logs and models
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = "./saved_models/train_{0}".format(str_time)
    os.makedirs(save_path)
    logging.info(f"Created directory for saving models and logs at: {save_path}")

    # Determine training stages
    if curriculum_parameters.get("enabled", False):
        training_stages = curriculum_parameters["stages"]
    else:
        # Fallback to default behavior if curriculum is not defined
        training_stages = [{
            "name": "Default Training",
            "num_jobs": env_parameters["num_jobs"],
            "num_mas": env_parameters["num_mas"],
            "is_dynamic": True, # Assuming dynamic is the default now
            "iterations": train_parameters["max_iterations"]
        }]

    total_iterations_done = 0
    for stage in training_stages:
        logging.info(f"--- Starting Curriculum Stage: {stage['name']} ---")
        
        # Update parameters for the current stage
        stage_env_params = copy.deepcopy(env_parameters)
        stage_env_params["num_jobs"] = stage["num_jobs"]
        stage_env_params["num_mas"] = stage["num_mas"]
        
        num_jobs = stage["num_jobs"]
        num_machines = stage["num_mas"]
        is_dynamic = stage["is_dynamic"]
        stage_iterations = stage["iterations"]

        # training loop for the current stage
        env_training = None
        for i in range(1, stage_iterations + 1):
            current_iteration = total_iterations_done + i
            if (i - 1) % train_parameters["parallel_iter"] == 0:
                nums_ope = [random.randint(int(num_machines * 0.8), int(num_machines * 1.2)) for _ in range(num_jobs)]
                case = CaseGenerator(num_jobs, num_machines, int(num_machines * 0.8), int(num_machines * 1.2), nums_ope, is_dynamic=is_dynamic)
                env_training = FJSPEnv_training(case=case, env_paras=stage_env_params)

            env_training.reset()
            state, done = env_training.state, False
            dones = env_training.done_batch

            # Schedule in parallel
            while not done:
                with torch.no_grad():
                    actions = model.policy_old.act(state, memories, dones)
                state, rewards, dones, _ = env_training.step(actions)
                done = dones.all()
                memories.rewards.append(rewards)
                memories.is_terminals.append(dones)

            # Check and reset environment for next iteration
            if not env_training.validate_gantt()[0]:
                logging.warning("Scheduling error encountered during validation.")
            env_training.reset()

            # Update model periodically
            if current_iteration % train_parameters["update_timestep"] == 0:
                loss, reward = model.update(memories, stage_env_params, train_parameters)
                logging.info(f"Iteration {current_iteration}: Model updated. Reward: {reward:.3f}, Loss: {loss:.3f}")
                memories.clear_memory()
                if is_viz:
                    viz.line(X=np.array([current_iteration]), Y=np.array([reward]), win="reward_envs", update="append")
                    viz.line(X=np.array([current_iteration]), Y=np.array([loss]), win="loss_envs", update="append")
        
        total_iterations_done += stage_iterations


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    train_FJSP_DRL(**parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train FJSP_DRL")
    parser.add_argument(
        "config_file",
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
