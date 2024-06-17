import gymnasium as gym
from agent_template.agent_template import HumanoidTemplate

from utils.SimpleConverter import SimpleConverter
import yaml
from box import Box


def load_args():
    # Path to your YAML file
    yaml_file_path = 'config_params/punch.yaml'
    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        args = Box(yaml.safe_load(file))
    
    return args


def load_arguments():
    args = load_args()
    trajectory = SimpleConverter(args.ref)
    trajectory.load_mocap()
    
    return trajectory,args





if __name__ == '__main__':
    
    trajectory,args = load_arguments()
    
    env = HumanoidTemplate(model_path=args.model,reference_data=trajectory,args=args,
                           frame_skip=1, render_mode="human")
    
    
    # from gymnasium.utils.env_checker import check_env
    # check_env(env.unwrapped)
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    
    # Testing the custom environment
    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info     
        #print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        print("this is the reward:", reward)
        #print("this is the info:", info)
        #print("this is the pose error:", info['pose_error'])
        
        if terminated or truncated:
            #print(env.current_idx)
            observation, info = env.reset()
            #print('end')
            #break
    
    env.close()
    
    