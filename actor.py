import ray
import numpy as np
import torch
from torch.nn import L1Loss
from ddpg import ActorNet, CriticNet
import copy
from environment import Environment
from replay_buffer import LocalBuffer
from utils import LinearSchedule

import warnings
warnings.filterwarnings('ignore')


@ray.remote(num_cpus=0.5, num_gpus=0.12)
class Actor_DDPG(object):
    def __init__(self, id, shared_memory, storage, parameters):
        self.id = id
        self.shared_memory = shared_memory
        self.storage = storage
        self.parameters = parameters
        self.local_buffer = LocalBuffer(parameters)

        self.gamma = parameters['gamma']

        self.agent = DDPG_Agent(parameters['action_dim'], parameters['state_dim'], parameters)
        self.agent.actor.eval()
        self.agent.actor_target.eval()
        self.agent.critic.eval()
        self.agent.critic_target.eval()

        self.env = Environment()
        self.std_noise_schedule = LinearSchedule(parameters['training_iterations'], final_p=0.0, initial_p=0.3)
        self.epsilon_schedule = LinearSchedule(parameters['training_iterations'], final_p=0.0, initial_p=0.9)

        self.total_tranition = self.parameters['total_transitions'] // self.parameters['actor_num']
        self.last_model_index = -1


    def run(self):
        done = False
        obs = self.env.reset()
        time_step = 0
        episode_num = 0
        episode_reward = 0.
        episode_timesteps = 0
        episode_backtimes = 0
        noise_std = 0.
        log_count = 0
        state_mean, state_std, action_mean, action_std = ray.get(self.shared_memory.get_state_action_mean_std.remote())

        if self.parameters['load_model']:
            self.agent.load('./models/'+'lsh')

        while True:
            start_training = ray.get(self.shared_memory.get_start_signal.remote())
            train_step = ray.get(self.shared_memory.get_counter.remote())

            episode_timesteps += 1
            # if episode_timesteps % 10 == 0:
            #     print(f'step={episode_timesteps}')
            # Epsilon-Greedy
            action = self.agent.act(obs)
            if np.random.uniform(0, 1) < 0.1:
                action += np.random.normal(0, 1, (self.parameters['action_dim']))

            next_obs, reward, done, info = self.env.step(action)

            episode_reward += reward

            self.local_buffer.add(obs, action, next_obs, reward, float(done))

            obs = copy.copy(next_obs)

            new_model_index = train_step // self.parameters['actor_update_interval']
            if start_training and new_model_index > self.last_model_index:
                if self.id == 0:
                    print(f'=======actor model updated========\n new_model_index={new_model_index}')
                self.last_model_index = new_model_index
                actor_weights, actor_target_weights, critic_weights, critic_target_weights = ray.get(self.shared_memory.get_weights.remote())
                self.agent.actor.set_weights(actor_weights)
                self.agent.actor.cuda()
                self.agent.actor.eval()
                self.agent.actor_target.set_weights(actor_target_weights)
                self.agent.actor_target.cuda()
                self.agent.actor_target.eval()
                self.agent.critic.set_weights(critic_weights)
                self.agent.critic.cuda()
                self.agent.critic.eval()
                self.agent.critic_target.set_weights(critic_target_weights)
                self.agent.critic_target.cuda()
                self.agent.critic_target.eval()

            if done or ((episode_timesteps+1) % self.parameters['actor_buffer_size'] == 0 and time_step > 0):
                # state_mean, state_std, action_mean, action_std = ray.get(self.shared_memory.get_state_action_mean_std.remote())
                # state_mean_b = torch.from_numpy(state_mean).float().to(self.device)
                # state_std_b = torch.from_numpy(state_std).float().to(self.device)
                # action_mean_b = torch.from_numpy(action_mean).float().to(self.device)
                # action_std_b = torch.from_numpy(action_std).float().to(self.device)

                print(f'actor_id={self.id}, info={info}, episode_num={episode_num}, episode_reward={episode_reward:.2f}, episode_steps={episode_timesteps}')

                state_b, action_b, next_state_b, reward_b, done_b = self.local_buffer.get_whole_buffer()

                # Compute the target Q value using the information of next state
                next_action_b = self.agent.actor_target(next_state_b)
                # TODO: add action normalization
                Q_tmp = self.agent.critic_target(next_state_b, next_action_b)
                Q_target = reward_b + self.gamma * (1 - done_b) * Q_tmp
                Q_current = self.agent.critic(state_b, action_b)

                priorities = L1Loss(reduction='none')(Q_current, Q_target).data.cpu().numpy() + 1e-5
                # if info == {}:  # only full episodes could be sent to shared buffer
                print(f'----------------------------------- actor {self.id} push a full episode ---------------------------------')
                self.storage.push_trajectory(
                    (state_b.detach().cpu().numpy(),
                    action_b.detach().cpu().numpy(),
                    next_state_b.detach().cpu().numpy(),
                    reward_b.detach().cpu().numpy(),
                    done_b.detach().cpu().numpy(), priorities)
                )

                # Reset environment
                obs, done = self.env.reset(), False
                average_reward = episode_reward / episode_timesteps

                episode_num += 1

                # train_step = ray.get(self.shared_memory.get_counter.remote())
                if self.id == 0 and train_step > log_count*self.parameters['log_interval'] - 100 and train_step < (log_count+8)*self.parameters['log_interval']:
                    self.shared_memory.add_actor_log.remote(average_reward, episode_reward, episode_timesteps, noise_std, episode_backtimes)
                    log_count += 1

                episode_reward = 0
                episode_timesteps = 0
                episode_backtimes = 0

            time_step += 1


class DDPG_Agent:
    def __init__(
            self,
            action_dim,
            state_dim,
            parameters
        ):

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_shape = (-1, state_dim)

        self.parameters = parameters
        self.gamma = parameters['gamma']
        self.tau = parameters['tau']

        self.actor = ActorNet(state_dim, action_dim).cuda()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = CriticNet(state_dim, action_dim).cuda()
        self.critic_target = copy.deepcopy(self.critic)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_DDPG_actor"))
        self.actor_target.load_state_dict(torch.load(filename + "_DDPG_actor"))
        self.critic.load_state_dict(torch.load(filename + "_DDPG_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_DDPG_critic"))

    def act(self, obs):
        state = torch.from_numpy(obs).unsqueeze(0).float().cuda()
        action = self.actor(state).squeeze().detach().cpu().numpy()
        return action
