# shared across tasks
from plb.optimizer.optim import Adam
from plb.engine.taichi_env import TaichiEnv
from plb.config.default_config import get_cfg_defaults, CN

import os
import cv2
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)
import matplotlib.pyplot as plt
from plb.config import load
from tqdm.notebook import tqdm

task_name = 'try'
env_type = '_fixed'

# gripper_fixed.yml
cfg = load(f"../envs/gripper{env_type}.yml") 
print(cfg)
env = TaichiEnv(cfg, nn=False, loss=False)
env.initialize()
state = env.get_state()

env.set_state(**state)
taichi_env = env
print(env.renderer.camera_pos)
env.renderer.camera_pos[0] = 0.5
env.renderer.camera_pos[1] = 2.5
env.renderer.camera_pos[2] = 0.5
env.renderer.camera_rot = (1.57, 0.0)

env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])

env.render('plt')

action_dim = taichi_env.primitives.action_dim

cwd = os.getcwd()
root_dir = cwd + "/../.."
print(f'root: {root_dir}')

task_params = {
    "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
    "sample_radius": 0.4,
    "len_per_grip": 30,
    "len_per_grip_back": 10,
    "floor_pos": np.array([0.5, 0, 0.5]),
    "n_shapes": 3, 
    "n_shapes_floor": 9,
    "n_shapes_per_gripper": 11,
    "gripper_mid_pt": int((11 - 1) / 2),
    "gripper_rate_limits": np.array([0.14, 0.06]), # ((0.4 * 2 - (0.23)) / (2 * 30), (0.4 * 2 - 0.15) / (2 * 30)),
    "p_noise_scale": 0.01,
}

if env_type == '':
    task_params["p_noise_scale"] = 0.03

print(f'p_noise_scale: {task_params["p_noise_scale"]}')

def set_parameters(env: TaichiEnv, yield_stress, E, nu):
    env.simulator.yield_stress.fill(yield_stress)
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    env.simulator.mu.fill(_mu)
    env.simulator.lam.fill(_lam)


def update_camera(env):
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)
    env.render_cfg.defrost()
    env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.2)
    env.render_cfg.camera_rot_1 = (0.8, 0.)
    env.render_cfg.camera_pos_2 = (2.4, 2.5, 0.2)
    env.render_cfg.camera_rot_2 = (0.8, 1.8)
    env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0.2)
    env.render_cfg.camera_rot_3 = (0.8, -1.8)
    env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
    env.render_cfg.camera_rot_4 = (0.8, 3.14)


def update_primitive(env, prim1_list, prim2_list):
    env.primitives.primitives[0].set_state(0, prim1_list)
    env.primitives.primitives[1].set_state(0, prim2_list)


def save_files(env, rollout_dir, i):
    files = glob.glob(f"{root_dir}/dataset/{task_name}/{i:03d}/*")
    for f in files:
        os.remove(f)
        
    os.makedirs(f"{rollout_dir}/{i:03d}", exist_ok=True)
    with open(f"{rollout_dir}/{i:03d}"+"/cam_params.npy", 'wb') as f:
        ext1=env.renderer.get_ext(env.render_cfg.camera_rot_1, np.array(env.render_cfg.camera_pos_1))
        ext2=env.renderer.get_ext(env.render_cfg.camera_rot_2, np.array(env.render_cfg.camera_pos_2))
        ext3=env.renderer.get_ext(env.render_cfg.camera_rot_3, np.array(env.render_cfg.camera_pos_3))
        ext4=env.renderer.get_ext(env.render_cfg.camera_rot_4, np.array(env.render_cfg.camera_pos_4))
        intrinsic = env.renderer.get_int()
        cam_params = {'cam1_ext': ext1, 'cam2_ext': ext2, 'cam3_ext': ext3, 'cam4_ext': ext4, 'intrinsic': intrinsic}
        np.save(f, cam_params)


from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat
def random_rotate(mid_point, gripper1_pos, gripper2_pos, z_vec):
    mid_point = mid_point[:3]
    z_angle = np.random.uniform(0, np.pi)
    z_mat = axangle2mat(z_vec, z_angle, is_normalized=True)
    all_mat = z_mat
    quat = mat2quat(all_mat)
    return gripper1_pos, gripper2_pos, quat


def random_pose(task_name):
    p_noise_x = task_params["p_noise_scale"] * (np.random.randn() * 2 - 1)
    p_noise_z = task_params["p_noise_scale"] * (np.random.randn() * 2 - 1)
    if task_name == 'try' or task_name == 'ngrip_3d':
        p_noise = np.clip(np.array([p_noise_x, 0, p_noise_z]), a_min=-0.1, a_max=0.1)
    else:
        raise NotImplementedError
    
    new_mid_point = task_params["mid_point"][:3] + p_noise

    rot_noise = np.random.uniform(0, np.pi)

    x1 = new_mid_point[0] - task_params["sample_radius"] * np.cos(rot_noise)
    z1 = new_mid_point[2] + task_params["sample_radius"] * np.sin(rot_noise)
    x2 = new_mid_point[0] + task_params["sample_radius"] * np.cos(rot_noise)
    z2 = new_mid_point[2] - task_params["sample_radius"] * np.sin(rot_noise)
    y = new_mid_point[1]
    z_vec = np.array([np.cos(rot_noise), 0, np.sin(rot_noise)])
    if task_name == 'try':
        gripper1_pos = np.array([x1, y, z1])
        gripper2_pos = np.array([x2, y, z2])
        quat = np.array([1, 0, 0, 0])
    elif task_name == 'ngrip_3d':
        gripper1_pos, gripper2_pos, quat = random_rotate(new_mid_point, np.array([x1, y, z1]), np.array([x2, y, z2]), z_vec)
    else:
        raise NotImplementedError
    return np.concatenate([gripper1_pos, quat]), np.concatenate([gripper2_pos, quat]), rot_noise


def get_obs(env, n_particles, t=0):
    x = env.simulator.get_x(t)
    v = env.simulator.get_v(t)
    step_size = len(x) // n_particles
    return x[::step_size], v[::step_size]


def select_tool(env, width):
    env.primitives.primitives[0].r[None] = width
    env.primitives.primitives[1].r[None] = width

import glob
from datetime import datetime
from tqdm.notebook import tqdm

i = 0
n_vid = 5
suffix = ''
n_grips = 3
zero_pad = np.array([0,0,0])

time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
rollout_dir = f"{root_dir}/dataset/{task_name}{env_type}{suffix}_{time_now}"

while i < n_vid: 
    print(f"+++++++++++++++++++{i}+++++++++++++++++++++")
    env.set_state(**state)
    taichi_env = env    
    update_camera(env)
    set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2 # 300, 800, 0.2
    update_primitive(env, [0.3, 0.4, 0.5, 1, 0, 0, 0], [0.7, 0.4, 0.5, 1, 0, 0, 0])
    save_files(env, rollout_dir, i)
    action_dim = env.primitives.action_dim
    imgs = [] 
    true_idx = 0
    for k in range(n_grips):
        print(k)
        prim1, prim2, cur_angle = random_pose(task_name)
        update_primitive(env, prim1, prim2)
        if 'small' in suffix:
            tool_size = 0.025
        else:
            tool_size = 0.045
        select_tool(env, tool_size)
        
        gripper_rate_limit = [(task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][0] + 2 * tool_size)) / (2 * task_params['len_per_grip']),
                              (task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][1] + 2 * tool_size)) / (2 * task_params['len_per_grip'])]
        rate = np.random.uniform(*gripper_rate_limit)
        actions = []
        counter = 0 
        mid_point = (prim1[:3] + prim2[:3]) / 2
        prim1_direction = mid_point - prim1[:3]
        prim1_direction = prim1_direction / np.linalg.norm(prim1_direction)
        while counter < task_params["len_per_grip"]:
            prim1_action = rate * prim1_direction
            actions.append(np.concatenate([prim1_action/0.02, zero_pad, -prim1_action/0.02, zero_pad]))
            counter += 1
        counter = 0
        while counter < task_params["len_per_grip_back"]:
            prim1_action = -rate * prim1_direction
            actions.append(np.concatenate([prim1_action/0.02, zero_pad, -prim1_action/0.02, zero_pad]))
            counter += 1

        actions = np.stack(actions)
            
        for idx, act in enumerate(tqdm(actions, total=actions.shape[0])):
            env.step(act)
            obs = get_obs(env, 300)
            x = obs[0][:300]
            
            primitive_state = [env.primitives.primitives[0].get_state(0), env.primitives.primitives[1].get_state(0)]

            img = env.render_multi(mode='rgb_array', spp=3)
            rgb, depth = img[0], img[1]

            os.system('mkdir -p ' + f"{rollout_dir}/{i:03d}")
            
            for num_cam in range(4):
                cv2.imwrite(f"{rollout_dir}/{i:03d}/{true_idx:03d}_rgb_{num_cam}.png", rgb[num_cam][..., ::-1])
            with open(f"{rollout_dir}/{i:03d}/{true_idx:03d}_depth_prim.npy", 'wb') as f:
                np.save(f, depth + primitive_state + [tool_size])
            with open(f"{rollout_dir}/{i:03d}/{true_idx:03d}_gtp.npy", 'wb') as f:
                np.save(f, x)
            true_idx += 1

        print(true_idx)
    
    os.system(f'ffmpeg -y -i {rollout_dir}/{i:03d}/%03d_rgb_0.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {rollout_dir}/{i:03d}/vid{i:03d}.mp4')
    i += 1