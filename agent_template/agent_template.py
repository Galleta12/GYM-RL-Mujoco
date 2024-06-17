import os
import math
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle
import sys
import os
from copy import deepcopy
from .pd_controllers_agents import stable_pd_controller
from .losses import loss_l2_relpos
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.util_data import generate_kp_kd_gains
from some_math import rfcmath
from some_math import rfctransformations
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


class HumanoidTemplate(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, model_path, reference_data, args, 
                 frame_skip: int = 5,   
                 default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,**kwargs):
        self.metadata['render_fps'] = int(np.round(self.metadata['render_fps'] / frame_skip))

        print("this is the model_path",model_path)
        # Hard-coded values for the number of values in qpos (nq = 35) and qvel (nv = 34) and phi 1
        obs_space = gym.spaces.Dict({
            'agent': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float64)
        })

        MujocoEnv.__init__(
            self, os.path.abspath(model_path), frame_skip, observation_space=obs_space,  
            default_camera_config=DEFAULT_CAMERA_CONFIG,**kwargs
        )
        EzPickle.__init__(self,model_path,reference_data,args,frame_skip,
                           default_camera_config,
                          **kwargs)
        #refernece data
        self.reference_data = mj.MjData(self.model)
        #refernece to the model of the kinematics
        self.model_reference = deepcopy(self.model)
        
        self.reference_trajectory_qpos = np.asarray(reference_data.data_pos)
        self.reference_trajectory_qvel = np.asarray(reference_data.data_vel)
        
        self.kp_gains,self.kd_gains = generate_kp_kd_gains()
        
        #this it is the lenght, on args
        self.rollout_lenght = args.ep_len
        
        self.cycle_len = self.reference_trajectory_qpos.shape[0]
        
        self.dict_ee = np.array([6,9,12,15])  
        self.current_idx =0
        
        #initialize important data
        self.body_qposaddr= self.set_qposddr()
        self.body_names = self.set_body_names()
        #parameters of rewards
        self.w_p =  args.deep_mimic_reward_weights.w_p
        self.w_v =  args.deep_mimic_reward_weights.w_v
        self.w_e =  args.deep_mimic_reward_weights.w_e
        self.w_c= args.deep_mimic_reward_weights.w_c    
        
        print(f"Reward weights - w_p: {self.w_p}, w_v: {self.w_v}, w_e: {self.w_e}, w_c: {self.w_c}")
        
        
        self.w_pose =  args.deep_mimic_weights_factors.w_pose
        self.w_angular =  args.deep_mimic_weights_factors.w_angular
        self.w_efector =  args.deep_mimic_weights_factors.w_efector
        self.w_com= args.deep_mimic_weights_factors.w_com
        
        print(f"Deep Mimic weights - w_pose: {self.w_pose}, w_angular: {self.w_angular}, w_efector: {self.w_efector}, w_com: {self.w_com}")   
        try:
            self.mujoco_renderer.render(render_mode=kwargs['render_mode'])
        except KeyError:
            pass
        
    
    def set_qposddr(self):
        body_qposaddr = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i): (self.model.body_jntadr[i], self.model.body_jntadr[i] + self.model.body_jntnum[i])
            for i in range(self.model.nbody)
        }
        return body_qposaddr
        
    def set_body_names(self):
        body_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        return body_names
    
    def get_reference_state(self,step_index,model,data)->bool:
        
        #step_index = step_index % self.cycle_len
        
        ref_qp = self.reference_trajectory_qpos[step_index]
        ref_qv = self.reference_trajectory_qvel[step_index]
        #now I will return a state depending on the index and the reference trajectory
        data.qpos[:] = ref_qp
        data.qvel[:] = ref_qv
        
        mj.mj_forward(model,data)
        
        
    #this is called before reset model
    def _reset_simulation(self):
        MujocoEnv._reset_simulation(self)
        mj.mj_resetData(self.model, self.data)
        mj.mj_resetData(self.model_reference, self.reference_data)
        #select the start index
        #self.current_idx = np.random.randint(0, self.rollout_lenght)
        self.current_idx = 0
                 
        #print("start idx:", self.current_idx)
        return self._get_obs()

    
    
    def _get_obs(self):
        
        qvel = self.data.qvel
        qpos = self.data.qpos

        qvel[:3] = rfcmath.transform_vec(qvel[:3],qpos[3:7], 'root').ravel()

        obs = []

        obs.append(qpos[2:])

        obs.append(qvel)
    
        phi = (self.current_idx % self.cycle_len)/self.cycle_len
        
        obs.append([phi])
        
        return {
            'agent': np.concatenate(obs)
        }
        
    def reset_model(self):
       
        qpos = self.reference_trajectory_qpos[self.current_idx]
        qvel = self.reference_trajectory_qvel[self.current_idx]
        
        self.set_state(qpos, qvel)
        #initial position
        self.default_pos = self.data.qpos
        #save the quat at the begginnig is the same for both since it starts
        #on the same position
        observation = self._get_obs()
        return observation
 
    
    
    
    def step(self, action):
        
        #increment the index by 1
        self.current_idx += 1
        # forward kinematics of refernece data
        #prev quat
        prev_ref_quat = self.get_body_quat(self.reference_data)
        
        self.get_reference_state(self.current_idx,self.model_reference,
                                 self.reference_data)
        #quat after kinematics
        current_ref_quat = self.get_body_quat(self.reference_data)
        
        time = self.data.time
        dt = self.model.opt.timestep
        #print("time" ,time )
        #print(self.current_idx)
        #action scale
        #action_scale = action * np.pi * 1.2
        #action_target = self.default_pos[7:] + action_scale
        #get the torque
        torque = stable_pd_controller(action,self.model,self.data,
                                       self.data.qpos,self.data.qvel,self.kp_gains,self.kd_gains,
                                       dt,time)        
        # torque = stable_pd_controller(self.reference_data.qpos[7:],self.model,self.data,
        #                               self.data.qpos,self.data.qvel,self.kp_gains,self.kd_gains,
        #                               dt,time)        
        #for the data
        prev_quat = self.get_body_quat(self.data)
        #for testing
        #self.get_reference_state(self.current_idx,self.model,self.data)
        #print(self.current_idx)
        self._step_mujoco_simulation(torque, n_frames=self.frame_skip)
        current_quat = self.get_body_quat(self.data)
        
        
        observation = self._get_obs()
        fall=0.0
        fall = np.where(self.data.qpos[2] < 0.2, 1.0, 0.0)
        fall = np.where(self.data.qpos[2] > 1.7, 1.0, fall)
        
        #print('fall', fall)
        
        done = True if fall ==1.0 else False
        #aslo done if end of cycle
        done = True if self.current_idx == self.cycle_len-1 else done
        
        #get angular velocity for reward
        angular_vel = rfcmath.get_angvel_fd(prev_quat,current_quat,dt)
        ref_angular_vel = rfcmath.get_angvel_fd(prev_ref_quat,current_ref_quat,dt)
        
        
        reward, reward_info = self.deep_mimic_reward(self.data,self.reference_data,
                                        angular_vel,ref_angular_vel,current_quat,current_ref_quat)
        
        #print("this is the reward:", reward)
        #calculate the pose error for metrics
        global_pos_state = self.data.xpos[1:]
        global_pos_ref = self.reference_data.xpos[1:]
        pose_error=loss_l2_relpos(global_pos_state, global_pos_ref)
        
        info = {
            'reward_quat': reward_info['reference_quaternions'],
            'reward_vel': reward_info['reference_angular'],
            'reward_end': reward_info['reference_end_effector'],
            'reward_center': reward_info['reference_com'],
            'pose_error':pose_error,
            'reward_ep': reward
        }
        
        if self.render_mode == "human":
            self.render()
        
        
        return observation,reward,done,False,info
    
    
    
    
    def get_center_mass_pos(self,data):
        return data.subtree_com[0, :].copy()
    
    def get_end_effectors(self,data):
        return data.geom_xpos[self.dict_ee].copy()
    
    def get_body_quat(self,data):
        qpos = data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = rfctransformations.quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat.copy()


    
    
    def deep_mimic_reward(self,data,ref_data,angular_data,ref_angular,current_quat,ref_quat):
        
        
        
        #get end effector
        end_effector = self.get_end_effectors(data)
        ref_end_effector = self.get_end_effectors(ref_data)
        
        #print(ref_end_effector)
        #get com
        com = self.get_center_mass_pos(data)
        ref_com = self.get_center_mass_pos(ref_data)
        
        reward_tuple = {
            'reference_quaternions': (
                self.quat_diff(current_quat,ref_quat)  
            ),
            'reference_angular': (
                self.angular_diff(angular_data , ref_angular)  
            ),
            'reference_end_effector': (
                self.end_effector_diff(end_effector,ref_end_effector) 
            ),
            'reference_com': (
                self.com_diff(com, ref_com) 
            )   
        }
        
        reward_quat = self.w_p * reward_tuple['reference_quaternions'] 
        reward_ang = self.w_v *reward_tuple['reference_angular'] 
        reward_end = self.w_e *reward_tuple['reference_end_effector'] 
        reward_com = self.w_c *reward_tuple['reference_com'] 
        
        reward = reward_quat + reward_ang + reward_end + reward_com
        
        return reward, reward_tuple
        
        
        
    #both quat diif and angular diff are one dim array
    def quat_diff(self,current_quat,ref_quat):
        #pose reward deepmimic
        pose_diff =rfcmath.multi_quat_norm(rfcmath.multi_quat_diff(ref_quat,current_quat))
        pose_dist = np.linalg.norm(pose_diff,ord=2)
        pose_reward = np.exp(-self.w_pose * (pose_dist ** 2))
        return pose_reward
        
        
    def angular_diff(self,angular_data , ref_angular):
        
        vel_dist = np.linalg.norm(ref_angular - angular_data , ord=2)
        vel_reward = np.exp(-self.w_angular * (vel_dist ** 2))    
        
        return vel_reward
    
    
    def end_effector_diff(self,end_effector,ref_end_effector):
        ee_dist = np.linalg.norm(ref_end_effector - end_effector,ord=2,axis=1)
        ee_sum = np.sum(ee_dist**2)
        
        ee_reward = np.exp(-self.w_efector * (ee_sum))
        
        return ee_reward
    
    
    def com_diff(self,com, ref_com):
        com_dist = np.linalg.norm(ref_com - com,ord=2)
        com_reward = np.exp(-self.w_com * (com_dist))
        return com_reward
        
        
        
        
        