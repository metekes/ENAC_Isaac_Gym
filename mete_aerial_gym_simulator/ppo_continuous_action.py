# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import os
import random
import time

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import cv2
import math


from envs import *
from aerial_gym.utils import task_registry


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "quad_with_obstacles", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},

        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},

        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},

        # Algorithm specific arguments
        {"name": "--total-timesteps", "type":int, "default": 30000000,
            "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 0.0026,
            "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type":int, "default": 16,
            "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--anneal-lr", "action": "store_true", "default": False,
            "help": "Toggle learning rate annealing for policy and value networks"},
        {"name": "--gamma", "type":float, "default": 0.99,
            "help": "the discount factor gamma"},
        {"name": "--gae-lambda", "type":float, "default": 0.95,
            "help": "the lambda for the general advantage estimation"},
        {"name": "--num-minibatches", "type":int, "default": 2,
            "help": "the number of mini-batches"},
        {"name": "--update-epochs", "type":int, "default": 4,
            "help": "the K epochs to update the policy"},
        {"name": "--norm-adv-off", "action": "store_true", "default": False,
            "help": "Toggles advantages normalization"},
        {"name": "--clip-coef", "type":float, "default": 0.2,
            "help": "the surrogate clipping coefficient"},
        {"name": "--clip-vloss", "action": "store_true", "default": False,
            "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
        {"name": "--ent-coef", "type":float, "default": 0.0,
            "help": "coefficient of the entropy"},
        {"name": "--vf-coef", "type":float, "default": 2,
            "help": "coefficient of the value function"},
        {"name": "--max-grad-norm", "type":float, "default": 1,
            "help": "the maximum norm for the gradient clipping"},
        {"name": "--target-kl", "type":float, "default": None,
            "help": "the target KL divergence threshold"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args



class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, x, y):
        observations = self.env.reset(x, y)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action, x, y):
        observations, privileged_observations, rewards, dones, infos, frames = self.env.step(action, x, y)
        
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
            frames
        )
        


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs+2).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_obs+2).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, np.prod(envs.num_actions)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.num_actions)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":

    from potential_field_with_grid import potential_field_with_grid
    args = get_args()

    # Opencv DNN
    net = cv2.dnn.readNet("yolov4_tiny_custom_best.weights", "yolov4_tiny_custom_test.cfg")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.sim_device
    print("using device:", device)

    # env setup
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)

    envs = RecordEpisodeStatisticsTorch(envs, device)


    print("num actions: ",envs.num_actions)
    print("num obs: ", envs.num_obs)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    # load checkpoint if needed
    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, envs.num_obs+2), dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.num_actions), dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game

    _x = 240 * torch.ones((1, args.num_envs), device='cuda:0')
    _y = 180 * torch.ones((1, args.num_envs), device='cuda:0')
    x = torch.zeros((1, args.num_envs), device='cuda:0')
    y = torch.zeros((1, args.num_envs), device='cuda:0')
    height = 3 * torch.ones((1, args.num_envs), device='cuda:0')
    
    buffer_x = [[] for env_id in range(args.num_envs)]
    buffer_y = [[] for env_id in range(args.num_envs)]
    reset_flag = [True for env_id in range(args.num_envs)]
    _reset_flag = reset_flag

    buffer_size = 1

    global_step = 0
    start_time = time.time()
    next_obs,_info = envs.reset(x, y)
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size

    if not args.play:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = torch.cat((next_obs, torch.t(x), torch.t(y)), 1)
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(torch.cat((next_obs, torch.t(x), torch.t(y)), 1))
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                # Check for getting outo f env boundary and detecting landing point far away when roll angle is high
                x = torch.where(abs(x) > 15, 0, x)
                y = torch.where(abs(y) > 15, 0, y)
                next_obs, rewards[step], next_done, info, frames = envs.step(action, x, y)

                for reset_id, reset in enumerate(next_done.detach().cpu().numpy()):
                    if reset:
                        buffer_x[reset_id] = []
                        buffer_y[reset_id] = []
                        reset_flag[reset_id] = reset

                if step == 1 or any(_reset_flag) == True:
                    _x_previous = _x
                    _y_previous = _y

                    w = frames[0].shape[1]
                    h = frames[0].shape[0]

                    for env_id in range(args.num_envs):
                        if _reset_flag[env_id] == True or step == 1:

                            '''
                            if obs[env_id, 2] < 0.5: 
                                x[0, env_id] = obs[env_id, 0].cpu()
                                y[0, env_id] = obs[env_id, 1].cpu()
                                continue
                            '''

                            image = frames[env_id]
                            if len(frames) == 0: 
                                print('No frame')
                                continue
                            (_x[0, env_id], _y[0, env_id], image) = potential_field_with_grid(image[:, :, :3], model)
                            
                            '''
                            if env_id == 0:
                                # image = cv2.resize(image, (560,560))
                                cv2.imshow('frame', image)
                                # cv2.imwrite('sim_dataset'+ str(counter)+'.jpg', image)
                                cv2.waitKey(8)
                            '''

                            # (_x,_y) = potential_field_with_grid(image[:, :, :3])
                            
                            if _x[0, env_id] == -1: _x[0, env_id] = w/2
                            if _y[0, env_id] == -1: _y[0, env_id] = h/2

                            '''
                            if _x[0, env_id] == -1: _x[0, env_id] = _x_previous[0, env_id]
                            if _y[0, env_id] == -1: _y[0, env_id] = _y_previous[0, env_id]
                            '''
                            
                            quat = isaacgym.gymapi.Quat(next_obs[env_id, 3].cpu().item(), next_obs[env_id, 4].cpu().item(), -next_obs[env_id, 5].cpu().item(), next_obs[env_id, 6].cpu().item())
                            _height = isaacgym.gymapi.Vec3(0.0, 0.0, next_obs[env_id, 2].cpu().item())
                            height[0, env_id] = quat.rotate(_height).dot(isaacgym.gymapi.Vec3(0.0, 0.0, 1.0))
                            
                            horizontal_fov = math.radians(env_cfg.env.onboard_camera_hor_fov)
                            vertical_fov = 2 * math.atan(math.tan(horizontal_fov/2) * (h/w))         
                            x[0, env_id] = (_x[0, env_id]- w/2)/(w/2) * math.tan(horizontal_fov/2) * height[0, env_id]
                            y[0, env_id] = (-_y[0, env_id]+ h/2)/(h/2) * math.tan(vertical_fov/2) * height[0, env_id]
                            inv_quat = quat.inverse()
                            relative_coordinate = inv_quat.rotate(isaacgym.gymapi.Vec3(x[0, env_id], y[0, env_id], height[0, env_id]))
                            x[0, env_id] = relative_coordinate.x
                            y[0, env_id] = relative_coordinate.y
                            x[0, env_id] = x[0, env_id] + next_obs[env_id, 0].cpu()
                            y[0, env_id] = y[0, env_id] + next_obs[env_id, 1].cpu()

                            if next_obs[env_id, 2].cpu() > 0.3:
                                if len(buffer_x[env_id]) >= buffer_size: del buffer_x[env_id][0]
                                if len(buffer_y[env_id]) >= buffer_size: del buffer_y[env_id][0]
                                buffer_x[env_id].append(x[0, env_id])
                                buffer_y[env_id].append(y[0, env_id])
                                x[0, env_id] = sum(buffer_x[env_id]) / len(buffer_x[env_id])
                                y[0, env_id] = sum(buffer_y[env_id]) / len(buffer_y[env_id])
                                buffer_x[env_id][-1] = x[0, env_id]
                                buffer_y[env_id][-1] = y[0, env_id]

                _reset_flag = reset_flag
                reset_flag = [False for env_id in range(args.num_envs)]

                if 0 <= step <= 2:
                    for idx, d in enumerate(next_done):
                        if d:
                            episodic_return = info["r"][idx].item()
                            print(f"global_step={global_step}, episodic_return={episodic_return}")
                            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                                writer.add_scalar(
                                    "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                                )
                            break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(torch.cat((next_obs, torch.t(x), torch.t(y)), 1)).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1, envs.num_obs+2))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, envs.num_actions))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
            # save the model levery 50 updates
            if update % 30 == 0:
                print("Saving model.")
                torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pth")



    else:
        _x = 240 * torch.ones((1, args.num_envs), device='cuda:0')
        _y = 180 * torch.ones((1, args.num_envs), device='cuda:0')
        x = torch.zeros((1, args.num_envs), device='cuda:0')
        y = torch.zeros((1, args.num_envs), device='cuda:0')
        height = 3 * torch.ones((1, args.num_envs), device='cuda:0')
    
        buffer_x = [[] for env_id in range(args.num_envs)]
        buffer_y = [[] for env_id in range(args.num_envs)]
        reset_flag = [True for env_id in range(args.num_envs)]
        _reset_flag = reset_flag

        for step in range(0, 5000000):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(torch.cat((next_obs, torch.t(x), torch.t(y)), 1))

            x = torch.where(abs(x) > 15, 0, x)
            y = torch.where(abs(y) > 15, 0, y)
            next_obs, rewards, next_done, info, frames = envs.step(action, x, y)


            for reset_id, reset in enumerate(next_done.detach().cpu().numpy()):
                if reset:
                    buffer_x[reset_id] = []
                    buffer_y[reset_id] = []
                    reset_flag[reset_id] = reset

            if step == 1 or any(_reset_flag) == True:
                _x_previous = _x
                _y_previous = _y

                w = frames[0].shape[1]
                h = frames[0].shape[0]

                for env_id in range(args.num_envs):
                    if _reset_flag[env_id] == True or step == 1:

                        '''
                        if obs[env_id, 2] < 0.5: 
                            x[0, env_id] = obs[env_id, 0].cpu()
                            y[0, env_id] = obs[env_id, 1].cpu()
                            continue
                        '''

                        image = frames[env_id]
                        if len(frames) == 0: 
                            print('No frame')
                            continue
                        (_x[0, env_id], _y[0, env_id], image) = potential_field_with_grid(image[:, :, :3], model)
                        
                        

                        # (_x,_y) = potential_field_with_grid(image[:, :, :3])
                        
                        if _x[0, env_id] == -1: _x[0, env_id] = w/2
                        if _y[0, env_id] == -1: _y[0, env_id] = h/2

                        '''
                        if _x[0, env_id] == -1: _x[0, env_id] = _x_previous[0, env_id]
                        if _y[0, env_id] == -1: _y[0, env_id] = _y_previous[0, env_id]
                        '''
                        
                        
                        quat = isaacgym.gymapi.Quat(next_obs[env_id, 3].cpu().item(), next_obs[env_id, 4].cpu().item(), next_obs[env_id, 5].cpu().item(), next_obs[env_id, 6].cpu().item())
                        _height = isaacgym.gymapi.Vec3(0.0, 0.0, next_obs[env_id, 2].cpu().item())
                        height[0, env_id] = _height.dot(isaacgym.gymapi.Vec3(0.0, 0.0, 1.0))
                        
                        horizontal_fov = math.radians(env_cfg.env.onboard_camera_hor_fov)
                        vertical_fov = 2 * math.atan(math.tan(horizontal_fov/2) * (h/w))
                            
                        x[0, env_id] = (-_x[0, env_id] + w/2)/(w/2) * math.tan(horizontal_fov/2) * height[0, env_id]
                        y[0, env_id] = (-_y[0, env_id] + h/2)/(h/2) * math.tan(vertical_fov/2) * height[0, env_id]
                        inv_quat = quat.inverse()
                        relative_coordinate = quat.rotate(isaacgym.gymapi.Vec3(x[0, env_id], y[0, env_id], -height[0, env_id]))
                        x[0, env_id] = relative_coordinate.x
                        y[0, env_id] = relative_coordinate.y
                        x[0, env_id] = x[0, env_id] + next_obs[env_id, 0].cpu()
                        y[0, env_id] = y[0, env_id] + next_obs[env_id, 1].cpu()

                        if next_obs[env_id, 2].cpu() > 0.3:
                            if len(buffer_x[env_id]) >= buffer_size: del buffer_x[env_id][0]
                            if len(buffer_y[env_id]) >= buffer_size: del buffer_y[env_id][0]
                            buffer_x[env_id].append(x[0, env_id])
                            buffer_y[env_id].append(y[0, env_id])
                            x[0, env_id] = sum(buffer_x[env_id]) / len(buffer_x[env_id])
                            y[0, env_id] = sum(buffer_y[env_id]) / len(buffer_y[env_id])
                            buffer_x[env_id][-1] = x[0, env_id]
                            buffer_y[env_id][-1] = y[0, env_id]


                        if env_id == 0:

                            quat = isaacgym.gymapi.Quat(next_obs[env_id, 3].cpu().item(), next_obs[env_id, 4].cpu().item(), next_obs[env_id, 5].cpu().item(), next_obs[env_id, 6].cpu().item())
                            _height = isaacgym.gymapi.Vec3(0.0, 0.0, next_obs[env_id, 2].cpu().item())
                            height[0, env_id] = _height.dot(isaacgym.gymapi.Vec3(0.0, 0.0, 1.0))
                            
                            horizontal_fov = math.radians(env_cfg.env.onboard_camera_hor_fov)
                            vertical_fov = 2 * math.atan(math.tan(horizontal_fov/2) * (h/w))         

                            coordinate = inv_quat.rotate(isaacgym.gymapi.Vec3(x[0, env_id], y[0, env_id], height[0, env_id]))
                            __x = coordinate.x - next_obs[env_id, 0].cpu()
                            __y = coordinate.y - next_obs[env_id, 1].cpu()

                            __x = -(__x) * w/2 / (math.tan(horizontal_fov/2) * height[0, env_id]) + w/2
                            __y = -(__y) * h/2 / (math.tan(vertical_fov/2) * height[0, env_id]) + h/2

                            # print(__x, __y)
                            cv2.circle(image, (int(__x), int(__y)), 16, (200, 100, 214), -1)

                            # image = cv2.resize(image, (560,560))
                            cv2.imshow('frame', image)
                            # cv2.imwrite('sim_dataset'+ str(counter)+'.jpg', image)
                            cv2.waitKey(8)
            _reset_flag = reset_flag
            reset_flag = [False for env_id in range(args.num_envs)]


    # envs.close()
    writer.close()