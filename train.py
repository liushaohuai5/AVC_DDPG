import numpy as np
import torch
import time
import ray
from ddpg import ActorNet, CriticNet
from torch.nn import L1Loss
import copy
from collections import deque
import torch.nn.functional as F


# @ray.remote(num_gpus=0.3)
class Learner_DDPG(object):
    def __init__(self, id, shared_memory, storage, parameters, summary_writer):
        self.id = id
        self.shared_memory = shared_memory
        self.storage = storage
        self.parameters = parameters
        self.summary_writer = summary_writer

        self.time_step = 0

        self.gamma = parameters['gamma']
        self.tau = parameters['tau']

        self.actor = ActorNet(self.parameters['state_dim'], self.parameters['action_dim']).cuda()
        self.actor.train()
        self.critic = CriticNet(self.parameters['state_dim'], self.parameters['action_dim']).cuda()
        self.critic.train()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=parameters['actor']['lr'],
                                                betas=parameters['actor']['betas'],
                                                weight_decay=parameters['actor']['weight_decay']
                                                )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=parameters['critic']['lr'],
                                                 betas=parameters['actor']['betas'],
                                                 weight_decay=parameters['critic']['weight_decay']
                                                 )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.last_Q = deque(maxlen=10)

        self.model_saved_flag = False

    def adjust_lr(self, optimizer, decay_rate=0.5):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train(self, batch):
        # batch_item, state_mean, state_std, action_mean, action_std = batch
        batch_item, _, _, _, _ = batch
        state, action, next_state, reward, done, ind, weights_lst = batch_item


        state = torch.from_numpy(state).float().cuda()
        action = torch.from_numpy(action).float().float().cuda()
        next_state = torch.from_numpy(next_state).float().cuda()
        reward = torch.from_numpy(reward).float().cuda()
        done = torch.from_numpy(done).float().cuda()
        weights = torch.from_numpy(weights_lst).float().cuda()

        # state_mean = torch.from_numpy(state_mean).float().to(self.device)
        # state_std = torch.from_numpy(state_std).float().to(self.device)
        # action_mean = torch.from_numpy(action_mean).float().to(self.device)
        # action_std = torch.from_numpy(action_std).float().to(self.device)

        # Compute the target Q value using the information of next state
        next_action = self.actor_target(next_state)
        Q_next = self.critic_target(next_state, next_action)
        Q_target = reward + self.gamma ** (self.parameters['n_step_return']) * (1 - done) * Q_next
        Q_current = self.critic(state, action)

        # TODO: add PID controller intergral part to reduce Q value over-estimation, cumulative_errors += td_errors
        priorities = L1Loss(reduction='none')(Q_current, Q_target).data.cpu().numpy() + 1e-6
        self.storage.push_priority((ind, priorities))

        # Compute the current Q value and the loss
        td_errors = Q_target - Q_current
        critic_loss = torch.mean(weights * (td_errors ** 2))  # with importance sampling

        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()

        # Make action and evaluate its action values
        action_out = self.actor(state)
        Q = self.critic(state, action_out)
        actor_loss = -torch.mean(Q)
        # Optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optimizer.step()

        if self.time_step % 500 == 0:
            print(f'training steps={self.time_step}, Q={Q.mean():.3f}, actor loss={actor_loss:.3f}, '
                  f'critic loss={critic_loss:.3f}, batch_queue={self.storage.batch_queue.qsize()}')


        return {
            'training/Q': Q_current.mean().detach().cpu().numpy(),
            'training/target_Q': Q_target.mean().detach().cpu().numpy(),
            'training/critic_loss': critic_loss.mean().detach().cpu().numpy(),
            'training/actor_loss': actor_loss.mean().detach().cpu().numpy(),
            'training/lr': self.actor_optimizer.param_groups[0]['lr'],
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_DDPG_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_DDPG_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_DDPG_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_DDPG_critic_optimizer")

    def soft_target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_target_update(self):
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

    def _log(self, log_info):
        average_reward, episode_reward, episode_len, noise_std, episode_backtimes, test_max_score, test_mean_score, \
        Q, critic_loss, actor_loss, lr, total_transitions, train_steps = log_info
        if average_reward is not None:
            self.summary_writer.add_scalar('episode/average_reward', average_reward, train_steps)
            self.summary_writer.add_scalar('episode/cumulative_reward', episode_reward, train_steps)
            self.summary_writer.add_scalar('episode/total_steps', episode_len, train_steps)
            self.summary_writer.add_scalar('episode/backtimes', episode_backtimes, train_steps)
            self.summary_writer.add_scalar('statistics/std', noise_std, train_steps)
        if test_mean_score is not None:
            self.summary_writer.add_scalar('test/episodic_max_score', test_max_score, train_steps)
            self.summary_writer.add_scalar('test/episodic_mean_score', test_mean_score, train_steps)
        if Q is not None:
            self.summary_writer.add_scalar('training/Q', Q, train_steps)
            self.summary_writer.add_scalar('training/critic_loss', critic_loss, train_steps)
            self.summary_writer.add_scalar('training/actor_loss', actor_loss, train_steps)
            self.summary_writer.add_scalar('training/lr', lr, train_steps)
            self.summary_writer.add_scalar('training/total_transitions', total_transitions, train_steps)

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_DDPG_actor"))
        self.actor_target.load_state_dict(torch.load(filename + "_DDPG_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_DDPG_actor_optimizer"))
        self.critic.load_state_dict(torch.load(filename + "_DDPG_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_DDPG_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_DDPG_critic_optimizer"))

    def run(self):
        start_training = False
        while self.time_step < self.parameters['training_iterations']:
            batch = self.storage.pop_batch()
            if batch is None:
                # print(f'no batch...batchQ={self.storage.batch_queue.qsize()}')
                time.sleep(1)
                continue

            if not start_training:
                self.shared_memory.set_start_signal.remote()
                start_training = True
                print('=======================-----training begin-----=============================')

            train_info = self._train(batch)
            self.time_step += 1
            self.shared_memory.incr_counter.remote()

            self.soft_target_update()
            # if self.time_step % self.parameters['target_update_interval'] == 0:
            #     self.hard_target_update()
            #     print('target_model is updated')

            if self.time_step % self.parameters['actor_update_interval'] == 0:
                self.shared_memory.set_weights.remote(self.actor.get_weights(), self.actor_target.get_weights(),
                                                      self.critic.get_weights(), self.critic_target.get_weights())

            if self.time_step % self.parameters['log_interval'] == 0:
                self.shared_memory.add_learner_log.remote(train_info['training/Q'],
                                                          train_info['training/critic_loss'],
                                                          train_info['training/actor_loss'], train_info['training/lr'])
                log_info = ray.get(self.shared_memory.get_log.remote())
                self._log(log_info)

            if self.time_step % self.parameters['lr_decay_interval'] == 0:
                self.adjust_lr(self.actor_optimizer, self.parameters['lr_decay_rate'])
                self.adjust_lr(self.critic_optimizer, self.parameters['lr_decay_rate'])

            # time.sleep(0.5)
