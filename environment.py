import pandapower as pp
import numpy as np
import pandapower.networks as pn
import pandas as pd

class Environment:

    def __init__(self):
        self.net = pn.case33bw()
        self.load_ids = self.net.load.p_mw.keys().tolist()
        self.load_p_data = self.net.load.p_mw.values * self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/load_p.csv').values[:, self.load_ids])
        self.load_q_data = self.net.load.q_mvar.values * self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/load_q.csv').values[:, self.load_ids])
        self.max_renewable_p_data = self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/max_renewable_gen_p.csv').values) / 10
        self.init_net()
        self.len = self.load_p_data.shape[0]

        self.gen_ids = self.net.sgen.p_mw.keys().tolist()
        self.q_max = 0.1
        self.q_min = -0.1

    def normalize(self, data):
        return (data - data.min(0)) / (data.max(0) - data.min(0))

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
        pp.runpp(self.net, algorithm='fdbx', tolerance_mva=1e-3)
        v = self.net.res_bus.vm_pu.values
        q = self.net.res_bus.q_mvar.values
        p = self.net.res_bus.p_mw.values
        obs = np.concatenate((q, p, v))
        return obs

    def step(self, action):
        self.adjust_net(action)
        try:
            pp.runpp(self.net, algorithm='fdbx', tolerance_mva=1e-3)
            done = False
            v = self.net.res_bus.vm_pu.values
            q = self.net.res_bus.q_mvar.values
            p = self.net.res_bus.p_mw.values
            reward = self.calc_reward(v)
            info = {}
        except:
            done = True
            info = 'power flow not converged'
            v = np.zeros_like(self.net.res_bus.vm_pu.values)
            q = np.zeros_like(self.net.res_bus.q_mvar.values)
            p = np.zeros_like(self.net.res_bus.p_mw.values)
            reward = self.calc_reward(v)
        next_obs = np.concatenate((q, p, v))
        return next_obs, reward, done, info

    def adjust_net(self, action):
        self.net.sgen.q_mvar[self.gen_ids] += (action + 1) / 2 * (self.q_max - self.q_min) + self.q_min

    def calc_reward(self, voltages):
        reward = -np.abs(voltages - 1).sum()
        return reward