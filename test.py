import ray
import time
from actor import DDPG_Agent
from environment import Environment

@ray.remote(num_gpus=0.12)
class Testor(object):
    def __init__(self, parameters, shared_memory):
        self.agent = DDPG_Agent(parameters['action_dim'], parameters['state_dim'], parameters)
        self.agent.actor.eval()
        self.agent.actor_target.eval()
        self.agent.critic.eval()
        self.agent.critic_target.eval()
        self.shared_memory = shared_memory
        self.parameters = parameters

        self.test_count = 0
        self.last_test_index = 0

        if self.parameters['load_model']:
            self.agent.load('./models/'+'lsh')

    def run_task(self, actor_weights, state_mean, state_std):
        self.agent.actor.set_weights(actor_weights)
        self.agent.actor.to(self.device)
        self.agent.actor.eval()

        test_idx = [
            # 450, 500, 550, 600,
            22753, 16129, 74593, 45793, 32257, 53569, 13826,
            26785,
            1729, 17281,
            34273, 36289, 44353, 52417, 67105, 75169, 289, 4897, 15841, 31969,
            16980-144
        ]

        steps = []

        max_episode = len(test_idx)
        max_steps = 288
        episode_reward = [0 for _ in range(max_episode)]
        env = Environment()
        for episode in range(max_episode):
            print('------ episode ', episode)
            print(f'------ reset, start from {test_idx[episode]}')
            obs = env.reset(idx=test_idx[episode])

            done = False
            timestep = 0

            while timestep <= max_steps:
                print('------ step ', timestep)
                action, bias, recover_thermal_flag, action_ori = self.agent.act(obs)
                obs, reward, done, info = env.step(action)

                episode_reward[episode] += reward
                timestep += 1

                if done or timestep == max_steps:
                    print('info:', info)
                    print(f'episode cumulative reward={episode_reward[episode]}')

                    break

            steps.append(timestep)

        return max(episode_reward), sum(episode_reward) / len(episode_reward)

    def run(self):
        while True:
            step_count = ray.get(self.shared_memory.get_counter.remote())
            start_training = ray.get(self.shared_memory.get_start_signal.remote())
            new_test_index = step_count // self.parameters['test_interval']
            if start_training and new_test_index > self.last_test_index:
                self.last_test_index = new_test_index
                actor_weights, _, _, _ = ray.get(self.shared_memory.get_weights.remote())
                state_mean, state_std, action_mean, action_std = ray.get(self.shared_memory.get_state_action_mean_std.remote())
                test_max_score, test_mean_score = self.run_task(actor_weights, state_mean, state_std)

                self.shared_memory.add_test_log.remote(test_max_score, test_mean_score)
                self.test_count += 1
            else:
                time.sleep(10)