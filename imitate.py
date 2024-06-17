import gymnasium as gym
from agent_template.agent_template import HumanoidTemplate
from agent import ActorCriticPolicy
from utils.SimpleConverter import SimpleConverter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import yaml
from box import Box
import os
import argparse

# Create directories to hold models and logs
model_dir = "models_ppo"
log_dir = "logs_ppo"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)



# Function to create an environment instance
def make_env(model_path, reference_data, args, frame_skip, render_mode):
    def _init():
        return HumanoidTemplate(model_path=model_path, reference_data=reference_data, args=args, frame_skip=frame_skip, render_mode=render_mode)
    return _init


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

def train(env):
    # model = PPO(policy=ActorCriticPolicy, env=env, n_steps=1000, batch_size=256, n_epochs=8,
    #             gamma=0.95, tensorboard_log=log_dir, verbose=1, device='cuda')
    #vec_env = make_vec_env(make_env(model_path, reference_data, args, frame_skip, render_mode), n_envs=num_envs)
    
    
    model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        n_steps=4096,             # m = 4096 samples per policy update
        batch_size=256,           # n = 256 for minibatches
        n_epochs=8,               # Number of epochs per policy update
        gamma=0.95,               # Discount factor
        gae_lambda=0.95,          # Lambda for GAE
        clip_range=0.2,           # Likelihood ratio clipping
        ent_coef=0.0,             # Entropy coefficient (set to 0 if not specified)
        vf_coef=1.0,              # Value function coefficient
        max_grad_norm=0.5,        # Maximum gradient norm for clipping
        learning_rate=5e-5,       # Policy learning rate
        tensorboard_log=log_dir,
        verbose=1,
        device='cuda'
    )

    
    
    iters = 0
   
    while True:
        iters += 1
        TIMESTEPS = 1000
        model.learn(total_timesteps=TIMESTEPS, callback=TensorboardCallback(), progress_bar=True,reset_num_timesteps=False)
        
        model.save(f"{model_dir}/PPO_{TIMESTEPS*iters}")


def test(env,path_to_model):
    
    
    model = PPO.load(path_to_model)

    vec_env = model.get_env() if model.get_env() else make_vec_env(lambda: env)
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render('human')

    env.close()





class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('reward/quat', self.locals['infos'][0]['reward_quat'])
        self.logger.record('reward/velocity', self.locals['infos'][0]['reward_vel'])
        self.logger.record('reward/end', self.locals['infos'][0]['reward_end'])
        self.logger.record('reward/center', self.locals['infos'][0]['reward_center'])
        self.logger.record('reward/reward_ep', self.locals['infos'][0]['reward_ep'])
        self.logger.record('pose/error', self.locals['infos'][0]['pose_error'])

        return True


if __name__ == '__main__':
    trajectory,args = load_arguments()
    
    env = HumanoidTemplate(model_path=args.model,reference_data=trajectory,args=args,
                           frame_skip=1, render_mode="human")
    
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()
    
    
    
    
    
    if args.train:
        #train(args.model, trajectory, args, frame_skip=1, render_mode="human")
        train(env)
    if args.test:
        if os.path.isfile(args.test):
            test(env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')

    
    # from gymnasium.utils.env_checker import check_env
    # check_env(env.unwrapped)
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)

    
    
