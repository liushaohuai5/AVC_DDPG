import ray
from train import Learner_DDPG
from actor import Actor_DDPG, DDPG_Agent
from shared_memory import SharedMemory
from replay_buffer import LocalBuffer, SharedBuffer
from ddpg import ActorNet, CriticNet
from utils import Storage, LinearSchedule
from test import Testor
from environment import Environment
from torch.utils.tensorboard import SummaryWriter
import copy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parameters = {
        'algo': 'ddpg',
        'PER': True,
        'n_step_bootstrap': True,
        'n_step_return': 5,
        "start_timesteps": 600,
        "initial_eps": 0.9,
        "end_eps": 0.001,
        "eps_decay": 0.999,
        # Learning
        "gamma": 0.997,
        "init_temperature": 0.1,
        "batch_size": 512,
        "optimizer": "Adam",
        "alpha": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 5e-3
        },
        "actor": {
            "lr": 1e-4,
            "tau": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-4,
            "update_frequency": 1,
            "hidden_dim": 512,
            "hidden_depth": 2,
            "log_std_bound": [-5, 2]
        },
        "critic": {
            "lr": 2e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-4,
            "tau": 0.005,
            "target_update_frequency": 1,
            "hidden_dim": 512,
            "hidden_depth": 2
        },
        "tau": 0.001,
        "training_iterations": 1000 * 1000,
        "actor_buffer_size": 288,
        "shared_buffer_size": 1000 * 1000,
        "target_update_interval": 1000,
        "test_interval": 5e3,
        "only_power": True,
        "only_thermal": False,
        "guided_policy_search": True,
        "random_explore": 'Gaussian',  # Gaussian or EpsGreedy or none
        'actor_num': 8,
        'actor_update_interval': 1000,
        'log_interval': 1000,
        'total_transitions': 20 * 1000 * 1000,
        'encoder': 'mlp',
        'lr_decay_rate': 0.95,
        'lr_decay_interval': 10000,
        "learnable_temperature": True,
        'load_model': False
    }

    # get state dim and action dim
    env = Environment()
    obs = env.reset()
    parameters['state_dim'] = len(obs)
    parameters['action_dim'] = len(env.gen_to_bus)
    state_dim = parameters['state_dim']
    action_dim = parameters['action_dim']

    ray.init(num_gpus=2, num_cpus=80, object_store_memory=100 * 1024 * 1024 * 1024)

    summary_writer = SummaryWriter(comment="-VAC-converter")

    actor = ActorNet(state_dim, action_dim)
    critic = CriticNet(state_dim, action_dim)
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)

    storage = Storage()
    shared_memory = SharedMemory.remote(actor, actor_target, critic, critic_target, parameters)
    shared_buffer = SharedBuffer.remote(parameters, storage, shared_memory)

    actors = [Actor_DDPG.remote(i, shared_memory, storage, parameters) for i in
              range(parameters['actor_num'])]

    testor = Testor.remote(parameters, shared_memory)

    workers = []
    workers += [shared_buffer.run.remote()]
    workers += [actor.run.remote() for actor in actors]
    workers += [testor.run.remote()]

    learner = Learner_DDPG(0, shared_memory, storage, parameters, summary_writer)
    learner.run()

    print('finish')

