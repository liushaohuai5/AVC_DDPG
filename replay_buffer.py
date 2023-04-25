from collections import deque
import numpy as np
import torch
import ray

class LocalBuffer(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.state_dim = parameters['state_dim']
        self.num_actions = parameters['action_dim']
        # self.batch_size = parameters['batch_size']
        self.max_size = int(parameters['actor_buffer_size'])
        self.buffer_size = int(parameters['actor_buffer_size'])

        # n-steps bootstrapping
        self.n_steps = parameters['n_step_return']
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # Prioritized Experience Replay
        self.alpha = 0.6
        self.priorities = np.ones((self.max_size, 1))
        self.beta = 0.4

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, self.num_actions), dtype=np.float32)
        self.next_state = np.zeros_like(self.state)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        if self.parameters['n_step_bootstrap']:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            if len(self.n_step_buffer) == self.n_steps:
                state, action, reward, next_state, done = self.calc_n_step_return(self.n_step_buffer)

        max_prior = self.priorities[:self.crt_size].max() if self.crt_size > 0 else 1.0

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.priorities[self.ptr] = max_prior
        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def calc_n_step_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_steps):
            Return += (self.parameters['gamma'] ** idx) * n_step_buffer[idx][2]
        return n_step_buffer[0][0], n_step_buffer[0][1], \
               Return, \
               n_step_buffer[-1][3], n_step_buffer[-1][4]

    def get_whole_buffer(self):
        idx = self.ptr
        self.ptr = 0
        self.crt_size = 0
        self.n_step_buffer.clear()
        return (
            torch.from_numpy(self.state[:idx]).float().cuda(),
            torch.from_numpy(self.action[:idx]).float().cuda(),
            torch.from_numpy(self.next_state[:idx]).float().cuda(),
            torch.from_numpy(self.reward[:idx]).float().cuda(),
            torch.from_numpy(self.done[:idx]).float().cuda()
        )


@ray.remote
class SharedBuffer(object):
    def __init__(self, parameters, storage, shared_memory):
        self.parameters = parameters
        self.state_dim = parameters['state_dim']
        self.num_actions = parameters['action_dim']
        self.batch_size = parameters['batch_size']
        self.mini_batch_size = parameters['actor_buffer_size']
        self.max_size = int(parameters['shared_buffer_size'])
        self.buffer_size = int(parameters['shared_buffer_size'])

        self.storage = storage
        self.shared_memory = shared_memory

        # Prioritized Experience Replay
        self.alpha = 0.6
        self.priorities = np.ones((self.max_size, 1))
        self.beta = -0.4
        # self.beta_schedule = LinearSchedule(parameters['total_transitions'], final_p=1.0, initial_p=self.init_beta)

        self.n_step_return = parameters['n_step_return']

        self.ptr = 0
        self.crt_size = 0
        self.count = 0
        self.push_cnt = 0

        self.state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, self.num_actions), dtype=np.float32)
        self.next_state = np.zeros_like(self.state)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add_trajectory(self, state, action, next_state, reward, done, priorities):
        self.mini_batch_size = len(reward)
        if self.ptr + self.mini_batch_size > self.max_size:
            ind = [i for i in range(self.ptr, self.max_size)] + [i for i in range(0, self.ptr + self.mini_batch_size - self.max_size)]
        else:
            ind = [i for i in range(self.ptr, self.ptr + self.mini_batch_size)]
        self.state[ind] = state
        self.action[ind] = action
        self.next_state[ind] = next_state
        self.reward[ind] = reward
        self.done[ind] = done
        self.priorities[ind] = priorities

        self.ptr = (self.ptr + self.mini_batch_size) % self.max_size
        print(f'ptr={self.ptr}, len={len(reward)}')
        self.crt_size = min(self.crt_size + self.mini_batch_size, self.max_size)
        self.count += self.mini_batch_size
        self.shared_memory.incr_transitions.remote(self.count)

    def sample_batch(self):
        probs = self.priorities[:self.crt_size] ** self.alpha
        probs /= probs.sum()
        probs = np.squeeze(probs)
        if self.parameters['PER']:
            ind = np.random.choice(self.crt_size, self.batch_size, p=probs, replace=False)
        else:
            ind = np.random.choice(self.crt_size, self.batch_size, replace=False)

        weights_lst = (self.crt_size * probs[ind]) ** (-self.beta)
        weights_lst /= weights_lst.max()

        return (
            self.state[ind], self.action[ind],
            self.next_state[ind],
            self.reward[ind], self.done[ind],
            ind, weights_lst,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for i in range(len(batch_indices)):
            idx, prior = batch_indices[i], batch_priorities[i]
            self.priorities[idx] = prior

    def run(self):
        # start_training = False
        state_mean, state_std = None, None
        action_mean, action_std = None, None
        while True:
            trajectory = self.storage.pop_trajectory()
            if trajectory is not None:
                state, action, next_state, reward, done, origin_priorities = trajectory
                self.add_trajectory(state, action, next_state, reward, done, origin_priorities)

            priorities = self.storage.pop_priority()
            if priorities is not None:
                ind, priors = priorities
                self.update_priorities(ind, priors)

            if self.crt_size > self.batch_size:
                if self.push_cnt % 100 == 0:
                    state_mean, state_std = self.state[:self.crt_size].mean(axis=0), self.state[:self.crt_size].std(axis=0)
                    action_mean, action_std = self.action[:self.crt_size].mean(axis=0), self.action[:self.crt_size].std(axis=0)
                    self.shared_memory.set_state_action_mean_std.remote(state_mean, state_std, action_mean, action_std)
                self.storage.push_batch((self.sample_batch(), state_mean, state_std, action_mean, action_std))
                self.push_cnt += 1
                # time.sleep(0.5)
                # print(f'push 1 batch, batchQ={self.storage.batch_queue.qsize()}')