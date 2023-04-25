import ray
import torch
import numpy as np

@ray.remote
class SharedMemory(object):
    def __init__(self, actor, actor_target, critic, critic_target, parameters):
        self.step_counter = 0
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.average_reward_log = []
        self.episode_reward_log = []
        self.episode_backtimes_log = []
        self.episode_len_log = []
        self.noise_std_log = []
        self.test_max_score_log = []
        self.test_mean_score_log = []
        self.Q_log = []
        self.critic_loss_log = []
        self.actor_loss_log = []
        self.lr_log = []
        self.start = False

        self.score = 0
        self.total_transitions = 0

        self.parameters = parameters
        self.state_mean = np.zeros(parameters['state_dim'], dtype=np.float32)
        self.state_std = np.ones_like(self.state_mean)
        self.action_mean = np.zeros(parameters['action_dim'], dtype=np.float32)
        self.action_std = np.ones_like(self.action_mean)

        # if self.parameters['load_model']:
        #     self.load('./models/'+'lsh')

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_DDPG_actor"), map_location=torch.device('cpu'))
        self.actor_target.load_state_dict(torch.load(filename + "_DDPG_actor"), map_location=torch.device('cpu'))
        self.critic.load_state_dict(torch.load(filename + "_DDPG_critic"), map_location=torch.device('cpu'))
        self.critic_target.load_state_dict(torch.load(filename + "_DDPG_critic"), map_location=torch.device('cpu'))

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        return (self.actor.get_weights(), self.actor_target.get_weights(), self.critic.get_weights(), self.critic_target.get_weights())

    def set_weights(self, actor_weights, actor_target_weights, critic_weights, critic_target_weights):
        self.actor.set_weights(actor_weights)
        self.actor_target.set_weights(actor_target_weights)
        self.critic.set_weights(critic_weights)
        self.critic_target.set_weights(critic_target_weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def add_actor_log(self, average_reward, episode_reward, episode_len, noise_std, episode_backtimes):
        self.average_reward_log.append(average_reward)
        self.episode_reward_log.append(episode_reward)
        self.episode_len_log.append(episode_len)
        self.noise_std_log.append(noise_std)
        self.episode_backtimes_log.append(episode_backtimes)

    def add_test_log(self, test_max_score, test_mean_score):
        self.test_max_score_log.append(test_max_score)
        self.test_mean_score_log.append(test_mean_score)

    def add_learner_log(self, Q, critic_loss, actor_loss, lr):
        self.Q_log.append(Q)
        self.critic_loss_log.append(critic_loss)
        self.actor_loss_log.append(actor_loss)
        self.lr_log.append(lr)

    def get_log(self):
        average_reward = None if len(self.average_reward_log) == 0 else self.average_reward_log.pop()
        episode_reward = None if len(self.episode_reward_log) == 0 else self.episode_reward_log.pop()
        episode_len = None if len(self.episode_len_log) == 0 else self.episode_len_log.pop()
        noise_std = None if len(self.noise_std_log) == 0 else self.noise_std_log.pop()
        episode_backtimes = None if len(self.episode_backtimes_log) == 0 else self.episode_backtimes_log.pop()
        test_max_score = None if len(self.test_max_score_log) == 0 else self.test_max_score_log.pop()
        test_mean_score = None if len(self.test_mean_score_log) == 0 else self.test_mean_score_log.pop()
        Q = None if len(self.Q_log) == 0 else self.Q_log.pop()
        critic_loss = None if len(self.critic_loss_log) == 0 else self.critic_loss_log.pop()
        actor_loss = None if len(self.actor_loss_log) == 0 else self.actor_loss_log.pop()
        lr = None if len(self.lr_log) == 0 else self.lr_log.pop()
        total_transitions = self.total_transitions
        train_steps = self.step_counter

        return [average_reward, episode_reward, episode_len, noise_std, episode_backtimes, test_max_score, test_mean_score,
                Q, critic_loss, actor_loss, lr, total_transitions, train_steps]

    def get_mean_score(self):
        if len(self.test_mean_score_log) > 0:
            self.score = sum(self.test_mean_score_log)/len(self.test_mean_score_log)
        return self.score

    def incr_transitions(self, count):
        self.total_transitions = count

    def get_transitions(self):
        return self.total_transitions

    def set_state_action_mean_std(self, state_mean, state_std, action_mean, action_std):
        self.state_mean, self.state_std = state_mean, state_std
        self.action_mean, self.action_std = action_mean, action_std

    def get_state_action_mean_std(self):
        return self.state_mean, self.state_std, self.action_mean, self.action_std