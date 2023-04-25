import pandapower as pp
import numpy as np
import pandapower.networks as pn
import pandas as pd

class Environment:

    def __init__(self, is_test=False):
        self.init_net()
        self.gen_p_data = pd.read_csv('/workspace/AdversarialGridZero/data/gen_p.csv').values
        self.gen_q_data = pd.read_csv('/workspace/AdversarialGridZero/data/gen_q.csv').values
        self.load_p_data = pd.read_csv('/workspace/AdversarialGridZero/data/load_p.csv').values
        self.load_q_data = pd.read_csv('/workspace/AdversarialGridZero/data/load_q.csv').values
        self.max_renewable_p_data = pd.read_csv('/workspace/AdversarialGridZero/data/max_renewable_gen_p.csv').values
        self.len = self.gen_p_data.shape[0]

        self.load_ids = self.net.load.p_mw.keys().tolist()
        self.gen_ids = self.net.gen.p_mw.keys().tolist()
        self.q_max = 180
        self.q_min = -180

    def init_net(self):
        self.net = pn.case33bw()
        self.init_gen()

    def init_gen(self):
        for i in range(self.max_renewable_p_data.shape[1]):
            pp.create.create_sgen(self.net, i, self.max_renewable_p_data[0, i])

    def reset(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, self.len)
        self.init_net()
        self.net.load.p_mw[self.load_ids] = self.load_p_data[idx, self.load_ids]
        self.net.load.q_mvar[self.load_ids] = self.load_q_data[idx, self.load_ids]
        self.net.sgen.p_mw[self.gen_ids] = self.max_renewable_p_data[idx, self.gen_ids]
        self.net.sgen.q_mvar[self.gen_ids] = 0
        pp.runpp(self.net)
        v = self.net.res_bus.vm_pu.values
        q = self.net.res_bus.q_mvar.values
        p = self.net.res_bus.p_mw.values
        obs = np.concatenate((q, p, v))
        return obs

    def step(self, action):
        self.adjust_net(action)
        pp.runpp(self.net)
        v = self.net.res_bus.vm_pu.values
        q = self.net.res_bus.q_mvar.values
        p = self.net.res_bus.p_mw.values
        reward = self.calc_reward(v)
        next_obs = np.concatenate((q, p, v))
        done = self.net.converged
        info = {}
        if done:
            info = 'power flow not converged'
        return next_obs, reward, done, info

    def adjust_net(self, action):
        self.net.sgen.q_mvar[self.gen_ids] += (action + 1) / 2 * (self.q_max - self.q_min) + self.q_min

    def calc_reward(self, voltages):
        reward = -np.abs(voltages - 1).sum()
        return reward