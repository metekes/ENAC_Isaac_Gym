# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym import AERIAL_GYM_ROOT_DIR
import os

import isaacgym
from envs import *
from aerial_gym.utils import  get_args, task_registry, Logger

import numpy as np
import torch

import time
import random
import cv2
import math

def play(args):
    # global buffer_x, buffer_y
    
    # Opencv DNN
    net = cv2.dnn.readNet("yolov4_tiny_custom_best.weights", "yolov4_tiny_custom_test.cfg")
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    env_cfg = task_registry.get_cfgs(name=args.task)
    
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.control.controller = "lee_velocity_control"

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = 800 # number of steps before plotting states
    counter = 0

    actions = torch.zeros(env.num_envs, 4, device=env.device)
    obs = torch.zeros(env.num_envs, 4, device=env.device)
    _x = 240 * torch.ones((1, env.num_envs), device=env.device)
    _y = 180 * torch.ones((1, env.num_envs), device=env.device)
    x = torch.zeros((1, env.num_envs), device=env.device)
    y = torch.zeros((1, env.num_envs), device=env.device)
    height = 3 * torch.ones((1, env.num_envs), device=env.device)
    
    buffer_x = [[] for env_id in range(env.num_envs)]
    buffer_y = [[] for env_id in range(env.num_envs)]

    buffer_size = 5

    env.reset(x, y)

    for i in range(10*int(env.max_episode_length)):
        if counter == 0:
            start_time = time.time()
        counter += 1

        actions[:, 0] = torch.clip((x - obs[:, 0]) * 0.4, -1, 1)
        actions[:, 1] = torch.clip((y - obs[:, 1]) * 0.4, -1, 1)
        # actions[:, 2] = np.clip((5.0 - obs[:, 2]) * 0.2, -0.6, 0.6)
        actions[:, 3] = 0

        for env_id in range(env.num_envs):
            if (abs(x[0, env_id] - obs[env_id, 0]) < 1.2 * height[0, env_id] and abs(y[0, env_id] - obs[env_id, 1]) < 1.2 * height[0, env_id]):
                actions[env_id, 2] = torch.clip((-2 * obs[env_id, 2]) * 0.2, -0.5, 0.5)
                if obs[env_id, 2].item() < 0.3:
                    actions[env_id, 0] = torch.clip((x[0, env_id] - obs[env_id, 0]) * 0.2, -0.4, 0.4)
                    actions[env_id, 1] = torch.clip((y[0, env_id] - obs[env_id, 1]) * 0.2, -0.4, 0.4)
            else: actions[env_id, 2] = torch.clip((5.0 - obs[env_id, 2]) * 0.2, -0.5, 0.5)


        obs, priviliged_obs, rewards, resets, extras, images= env.step(actions.detach(), x, y)

        for reset_id, reset in enumerate(resets.detach().cpu().numpy()):
            if reset:
                buffer_x[reset_id] = []
                buffer_y[reset_id] = []

        if counter % 800 == 0:
            env.reset(x, y)
            end_time = time.time()
            print(f"FPS: {env_cfg.env.num_envs * 100 / (end_time - start_time)}")
            counter = 0

        
        if counter % 7 == 1 and env_cfg.env.enable_onboard_cameras:
            _x_previous = _x
            _y_previous = _y

            w = images[0].shape[1]
            h = images[0].shape[0]

            for env_id in range(env.num_envs):

                '''
                if obs[env_id, 2] < 0.5: 
                    x[0, env_id] = obs[env_id, 0].cpu()
                    y[0, env_id] = obs[env_id, 1].cpu()
                    continue
                '''

                image = images[env_id]
                if len(images) == 0: 
                    print('No frame')
                    continue
                (_x[0, env_id], _y[0, env_id], image) = potential_field_with_grid(image[:, :, :3], env_id, model)
                
                if env_id == 0:
                    # image = cv2.resize(image, (560,560))
                    cv2.imshow('frame', image)
                    # cv2.imwrite('sim_dataset'+ str(counter)+'.jpg', image)
                    cv2.waitKey(8)

                # (_x,_y) = potential_field_with_grid(image[:, :, :3])
                
                if _x[0, env_id] == -1: _x[0, env_id] = _x_previous[0, env_id]
                if _y[0, env_id] == -1: _y[0, env_id] = _y_previous[0, env_id]
                
                quat = isaacgym.gymapi.Quat(obs[env_id, 3].cpu().item(), obs[env_id, 4].cpu().item(), obs[env_id, 5].cpu().item(), obs[env_id, 6].cpu().item())
                _height = isaacgym.gymapi.Vec3(0.0, 0.0, obs[env_id, 2].cpu().item())
                height[0, env_id] = quat.rotate(_height).dot(isaacgym.gymapi.Vec3(0.0, 0.0, 1.0))
                
                horizontal_fov = math.radians(env_cfg.env.onboard_camera_hor_fov)
                vertical_fov = 2 * math.atan(math.tan(horizontal_fov/2) * (h/w))         
                x[0, env_id] = _x[0, env_id]/w * math.tan(horizontal_fov/2) * height[0, env_id]
                y[0, env_id] = _y[0, env_id]/h * math.tan(vertical_fov/2) * height[0, env_id]
                inv_quat = quat.inverse()
                x[0, env_id] = inv_quat.rotate(isaacgym.gymapi.Vec3(x[0, env_id], 0.0, 0.0)).dot(isaacgym.gymapi.Vec3(1.0, 0.0, 0.0))
                y[0, env_id] = inv_quat.rotate(isaacgym.gymapi.Vec3(0.0, y[0, env_id], 0.0)).dot(isaacgym.gymapi.Vec3(0.0, 1.0, 0.0))
                x[0, env_id] = x[0, env_id] + obs[env_id, 0].cpu()
                y[0, env_id] = y[0, env_id] + obs[env_id, 1].cpu()

                if obs[env_id, 2].cpu() < 0.3: continue
                if len(buffer_x[env_id]) > buffer_size: del buffer_x[env_id][0]
                if len(buffer_y[env_id]) > buffer_size: del buffer_y[env_id][0]
                buffer_x[env_id].append(x[0, env_id])
                buffer_y[env_id].append(y[0, env_id])
                x[0, env_id] = sum(buffer_x[env_id]) / len(buffer_x[env_id])
                y[0, env_id] = sum(buffer_y[env_id]) / len(buffer_y[env_id])
                
                

                buffer_x[env_id][-1] = x[0, env_id]
                buffer_y[env_id][-1] = y[0, env_id]

        
        if counter > 5 and counter%5 == 1 and env_cfg.env.enable_save_onboard_cameras:
            for env_id, frame in enumerate(images):
                cv2.imwrite('sim_dataset/' + str(int(time.time())) + '_' + str(env_id) + '.jpg', frame)

        if i < stop_state_log:
            abs_vel = torch.norm(env.root_states[:, 7:10], dim=1)
            logger.log_states(
                {
                    'command_action_x_vel': actions[robot_index, 0].item(),
                    'command_action_y_vel': actions[robot_index, 1].item(),
                    'command_action_z_vel': actions[robot_index, 2].item(),
                    'command_action_yaw_vel': actions[robot_index, 3].item(),
                    'reward': rewards[robot_index].item(),
                    'pos_x' : obs[robot_index, 0].item(),
                    'pos_y' : obs[robot_index, 1].item(),
                    'pos_z' : obs[robot_index, 2].item(),
                    'linvel_x': obs[robot_index, 7].item(),
                    'linvel_y': obs[robot_index, 8].item(),
                    'linvel_z': obs[robot_index, 9].item(),
                    'angvel_x': obs[robot_index, 10].item(),
                    'angvel_y': obs[robot_index, 11].item(),
                    'angvel_z': obs[robot_index, 12].item(),
                    'abs_linvel': abs_vel[robot_index].item()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()




def potential_field_with_grid(frame, env_id, model):
    grid_num = 31

    # Load class lists
    classes = ["occupied"]

    buffer_size = 10
    potential_field = np.zeros((grid_num, grid_num))

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.75, nmsThreshold=.7)
    obj = []

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if len(bboxes) == 0:
        frame = cv2.putText(frame, "Land Here", (frame.shape[1]//2 - 50, frame.shape[0]//2  - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 214), 2)
        frame = cv2.circle(frame, (frame.shape[1]//2 , frame.shape[0]//2 ), 5, (250, 0, 214), -1)
        return (-1, -1, frame)
    
    for class_id, score, bbox in zip(class_ids, scores, bboxes):

        (x_box, y_box, w, h) = bbox
        frame = cv2.rectangle(frame, (x_box, y_box), (x_box+w, y_box+h), (100, 110, 100), 3)
        
        x_center = (x_box + w//2)
        y_center = (y_box + h//2)

        obj.append((x_center, y_center, w, h))

        frame = cv2.circle(frame, (x_center, y_center), 5, (100, 110, 100), -1)

        for i in list(range(grid_num)):
            for j in list(range(grid_num)):
                x = i * frame.shape[1]/(grid_num - 1)
                y = j * frame.shape[0]/(grid_num - 1)
                
                try:
                    # if (x - x_center)/frame.shape[1] < 0.1 and (y - y_center)/frame.shape[0] < 0.1:
                    potential_field[j ,i] -= 0.5 * (((x - x_center)/w)**2 + ((y - y_center)/h)**2) ** 0.5
                        
                except:
                    potential_field[j ,i] += 10000


    c = np.max(np.abs(potential_field)) * 10
    # print(c)

    # print(potential_field)

    for i in range((len(range(grid_num)) - 1 )// 6):
        potential_field[i, :] += (10 - i) * c
        potential_field[-1-i, :] += (10 - i) * c
    for j in range((len(range(grid_num)) - 1) // 6):
        potential_field[:, j] += (10 - j) * c
        potential_field[:, -1-j] += (10 - j) * c
        # potential_field[i ,j] += math.sqrt((x - frame.shape[1]//2)**2 + ((y - frame.shape[0]//2)**2)) * 0.03

    for i in list(range(grid_num)):
        for j in list(range(grid_num)):
            x = i * frame.shape[1]/(grid_num - 1)
            y = j * frame.shape[0]/(grid_num - 1)

            for (x_center, y_center, w, h) in obj:

                if (x_center - w//2) <= x <= (x_center + w//2) and (y_center - h//2) <= y <= (y_center + h//2):
                    potential_field[j, i] += 10000

                if ((x - x_center)**2 + (y - y_center)**2)**0.5 < 1.3 * (w + h)/2:
                    potential_field[j, i] += 8000

    #  print(potential_field)

    min_index = np.unravel_index(potential_field.argmin(), potential_field.shape)
    min_y = int(min_index[0] * frame.shape[0]/(grid_num - 1))
    min_x = int(min_index[1] * frame.shape[1]/(grid_num - 1))
    


    '''
    if len(buffer_x[env_id]) > buffer_size: del buffer_x[env_id][0]
    if len(buffer_y[env_id]) > buffer_size: del buffer_y[env_id][0]
    buffer_x[env_id].append(min_x)
    buffer_y[env_id].append(min_y)
    min_x_ave = int(sum(buffer_x[env_id]) / len(buffer_x[env_id]))
    min_y_ave = int(sum(buffer_y[env_id]) / len(buffer_y[env_id]))
    
    for (x_center, y_center, w, h) in obj:
        if (x_center - w//2) <= min_x_ave <= (x_center + w//2) and (y_center - h//2) <= min_y_ave <= (y_center + h//2):
            min_x_ave = min_x
            min_y_ave = min_y
        elif ((min_x_ave - x_center)**2 + (min_y_ave - y_center)**2)**0.5 < 1.1 * (w + h)/2:
            min_x_ave = min_x
            min_y_ave = min_y

    buffer_x[env_id][-1] = min_x_ave
    buffer_y[env_id][-1] = min_y_ave
    
    '''


    frame = cv2.circle(frame, (min_x, min_y), 5, (250, 0, 214), -1)
    frame = cv2.putText(frame, "Land Here", (min_x - 50, min_y- 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 214), 2)
    



    
    return (min_x, min_y, frame)





if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
