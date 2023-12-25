from isaacgym import gymapi ,gymtorch
import torch
from isaacgym.torch_utils import get_euler_xyz
import math
import time
import numpy as np
import cv2
import isaacgym
def rotate(quat, vect):
    q1 =quat.x
    q2 =quat.y
    q3 =quat.z
    q4 =quat.w


    rot_matrix = np.array([[q1**2-q2**2-q3**2+q4**2, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
                           [2*(q1*q2 - q3*q4), -q1**2+q2**2-q3**2+q4**2, 2*(q3*q2 + q1*q4)],
                           [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), -q1**2-q2**2+q3**2+q4**2]])
    

    return np.matmul(rot_matrix, vect)

net = cv2.dnn.readNet('yolov4_tiny_custom_best.weights' , 'yolov4_tiny_custom_test.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (320, 320), scale = 1/255)

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

asset_root = "../resources"
asset_file = "robots/quad/model.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.armature = 0.01
asset_options.collapse_fixed_joints = True
asset_options.flip_visual_attachments = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)
spacing = 2000.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

env = gym.create_env(sim, lower, upper, 10)

actor_handle = []
pose = gymapi.Transform()
pose.p = gymapi.Vec3(-5, -3, 10.0)
rad = math.pi/180 * -5
pose.r = gymapi.Quat(1 * math.sin(rad), 3 * math.sin(rad), 2 * math.sin(rad), math.cos(rad))
# q0 = torch.tensor([[-0.707107, 0.0, 0.0, 0.707107]],dtype=torch.float32, device="cuda:0")

actor_handle.append(gym.create_actor(env, asset, pose, "MyActor", 0, 1))




pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
# rad = math.pi/180 * 10
# pose.r = gymapi.Quat(0 * math.sin(rad), 0 * math.sin(rad), 1 * math.sin(rad), math.cos(rad))
pose.r = gymapi.Quat(0, 0, 0, 1)
box = gym.create_box(sim, 1, 1, 1, gymapi.AssetOptions())
box_handle = gym.create_actor(env, box, pose, "MyActorewr", 0, 2)


viewer = gym.create_viewer(sim, gymapi.CameraProperties())
onboard_camera_properties = gymapi.CameraProperties()
# onboard_camera_properties.width = 120 # automatically multiplies by 4,. I dont know why
# onboard_camera_properties.height = 320
onboard_camera_properties.width = onboard_camera_properties.height
onboard_camera_properties. horizontal_fov = 87
onboard_camera_properties.enable_tensors = True
camera_handle = gym.create_camera_sensor(env, onboard_camera_properties)
local_transform = gymapi.Transform()
local_transform.p = gymapi.Vec3(0, 0, 0)
local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(-90.0))

gym.attach_camera_to_body(camera_handle, env, actor_handle[0], local_transform, gymapi.FOLLOW_TRANSFORM)
gym.render_all_camera_sensors(sim)

gym.prepare_sim(sim)

_root_tensor = gym.acquire_actor_root_state_tensor(sim)
# wrap it in a PyTorch Tensor
root_tensor = gymtorch.wrap_tensor(_root_tensor)
print(root_tensor)
initial_state = root_tensor.clone()
initial_state[:, :3] = torch.tensor([0, 0, 10])

mass = 0
for body in gym.get_actor_rigid_body_properties(env, actor_handle[0]):
    mass = mass + body.mass

num_actor = gym.get_sim_actor_count(sim)
force = torch.zeros((num_actor, 3),dtype=torch.float32, device="cuda:0")
torque = torch.zeros((num_actor, 3),dtype=torch.float32, device="cuda:0")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")



while not gym.query_viewer_has_closed(viewer):
    force[0:2, :3] = torch.tensor([[0, 0, 9.81 * mass]],dtype=torch.float32, device="cuda:0")
    torque[0:2, :3] = torch.tensor([[0, 0, 0]],dtype=torch.float32, device="cuda:0")
    for evt in gym.query_viewer_action_events(viewer):
        initial_state = root_tensor.clone()
        quat_ned = gymapi.Quat()
        quat_ned.from_euler_zyx(10, 20 ,12)
        quat_enu = gymapi.Quat(quat_ned.y, quat_ned.x,  -quat_ned.z, quat_ned.w)
        quat_enu = torch.tensor([quat_enu.x, quat_enu.y, quat_enu.z, quat_enu.w], device = 'cuda:0')
        initial_state[:, 3:7] = quat_enu
        initial_state[:, :3] = torch.tensor([0, 0, 10])
        if evt.action == 'forward' and evt.value > 0:
            #torque[0, :3] = torch.tensor([[0, 0.1, 0]],dtype=torch.float32, device="cuda:0")
            force[0, :3] = torch.tensor([[10, 0, 9.81 * mass]],dtype=torch.float32, device="cuda:0")

        elif evt.action == 'backward' and evt.value > 0:
            #torque[0, :3] = torch.tensor([[0, -0.1, 0]],dtype=torch.float32, device="cuda:0")
            force[0, :3] = torch.tensor([[-10, 0, 9.81 * mass]],dtype=torch.float32, device="cuda:0")
    
        elif evt.action == 'right' and evt.value > 0:
            #torque[0, :3] = torch.tensor([[0.1, 0, 0]],dtype=torch.float32, device="cuda:0")
            force[0, :3] = torch.tensor([[0, -10, 9.81 * mass]],dtype=torch.float32, device="cuda:0")
    
        elif evt.action == 'left' and evt.value > 0:
            #torque[0, :3] = torch.tensor([[-0.1, 0, 0]],dtype=torch.float32, device="cuda:0")
            force[0, :3] = torch.tensor([[0, 10, 9.81 * mass]],dtype=torch.float32, device="cuda:0")
        elif evt.action == 'reset':
            j =torch.tensor([0], dtype=torch.int32, device = 'cuda:0')
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(initial_state), gymtorch.unwrap_tensor(j)  ,1)
            print(initial_state)

               
        torque[1, :3] = torch.tensor([[0, 0, 0]],dtype=torch.float32, device="cuda:0")
    gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(force), 
                                            gymtorch.unwrap_tensor(torque), gymapi.ENV_SPACE)
 
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    quat = root_tensor[:, 3:7]
    (r, p, y) = get_euler_xyz(quat)
    
    
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.render_all_camera_sensors(sim)

    gym.start_access_image_tensors(sim)
    _image = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    image = gymtorch.wrap_tensor(_image) # the depth of the imnage is 4, RGBA
    image = image.detach().cpu().numpy()
    image = cv2.resize(image, (720,720))
    class_ids, scores, bboxes = model.detect(image[:, :, :3], confThreshold = 0.80, nmsThreshold = 0.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.putText(image, str(score), (x-20, y-20),cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=(100, 110, 100), thickness=2)
        image = cv2.rectangle(image, (x, y), (x+h, y+w), (100, 110, 100), 3)
        x = x + w/2
        y = y + h/2

        quat = isaacgym.gymapi.Quat(root_tensor[0, 3].cpu().item(), root_tensor[0, 4].cpu().item(), root_tensor[0, 5].cpu().item(), root_tensor[0, 6].cpu().item())
        inv_quat = quat.inverse()
        height = isaacgym.gymapi.Vec3(0.0, 0.0, root_tensor[0, 2].cpu().item()).dot(isaacgym.gymapi.Vec3(0, 0, 1))
                            
        w = image.shape[0]
        h = image.shape[1]
        horizontal_fov = math.radians(87)
        vertical_fov = 2 * math.atan(math.tan(horizontal_fov/2) * (h/w))
        _y = (-x + w/2)/(w/2) * math.tan(horizontal_fov/2) * height
        _x = (-y + h/2)/(h/2) * math.tan(vertical_fov/2) * height

        print(33333333333333)
   
        
        #relative_coordinate = inv_quat.rotate(isaacgym.gymapi.Vec3(x, y, height))
        #x = relative_coordinate.x
        #y = relative_coordinate.y

        print(_x, _y)
        relative_coordinate = quat.rotate(isaacgym.gymapi.Vec3(_x, _y, -height))
        _x = relative_coordinate.x
        _y = relative_coordinate.y
        print(_x, _y)
        

        # relative_coordinate = rotate(inv_quat, np.array([x, y, height]).T)
        #x = relative_coordinate[0]
        #y = relative_coordinate[1]
        
        
        
        _x = _x + root_tensor[0, 0].cpu()
        _y = _y + root_tensor[0, 1].cpu()

        print(_x, _y)

        

    
    cv2.imshow('a', image)
    cv2.waitKey(10)
    gym.start_access_image_tensors(sim)
    
    gym.sync_frame_time(sim)
    gym.refresh_actor_root_state_tensor(sim)
    

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
