import pandapower as pp
import numpy as np
import pandapower.networks as pn
import pandas as pd

class Environment:

    def __init__(self, is_test=False):
        self.net = pn.case33bw()
        pp.runpp(self.net, algorithm='fdbx', tolerance_mva=1e-3)
        self.ori_voltages = self.net.res_bus.vm_pu.values
        self.gen_to_bus = {0: 10, 1: 15, 2: 22, 3: 24, 4: 27, 5: 30}
        self.load_to_bus = {0: 8, 1: 12, 2: 17, 3: 19, 4: 25, 5: 28, 6: 30}
        self.ori_load_p = self.net.load.p_mw[list(self.load_to_bus.keys())].values
        self.ori_load_q = self.net.load.q_mvar[list(self.load_to_bus.keys())].values
        self.load_p_data = self.ori_load_p * self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/load_p.csv').values[:, list(self.load_to_bus.values())])
        self.load_q_data = self.ori_load_q * self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/load_q.csv').values[:, list(self.load_to_bus.values())])
        self.max_renewable_p_data = self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/max_renewable_gen_p.csv').values) / 10
        self.init_net()
        self.len = self.load_p_data.shape[0]
        self.idx = 0

        self.q_max = 0.1
        self.q_min = -0.1
        self.is_test = is_test

    def normalize(self, data):
        return (data - data.min(0)) / (data.max(0) - data.min(0))

    def init_net(self):
        self.net = pn.case33bw()
        self.init_gen()
        self.init_load()

    def init_gen(self):
        for i in self.gen_to_bus.keys():
            bus = self.gen_to_bus[i]
            pp.create.create_sgen(self.net, bus, self.max_renewable_p_data[0, i])

    def init_load(self):
        ori_ids = self.net.load.p_mw.keys().tolist()
        for i in ori_ids:
            self.net.load.drop(i)
        for i in self.load_to_bus.keys():
            bus = self.load_to_bus[i]
            pp.create.create_load(self.net, bus, self.load_p_data[0, i], self.load_q_data[0, i])

    def reset(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, self.len)
            self.idx = idx
        self.init_net()
        self.net.load.p_mw[list(self.load_to_bus.keys())] = self.load_p_data[idx, list(self.load_to_bus.keys())]
        self.net.load.q_mvar[list(self.load_to_bus.keys())] = self.load_q_data[idx, list(self.load_to_bus.keys())]
        self.net.sgen.p_mw[list(self.gen_to_bus.keys())] = self.max_renewable_p_data[idx, list(self.gen_to_bus.keys())]
        self.net.sgen.q_mvar[list(self.gen_to_bus.keys())] = 0
        pp.runpp(self.net, algorithm='fdbx', tolerance_mva=1e-3)
        v = self.net.res_bus.vm_pu.values
        q = self.net.res_bus.q_mvar.values
        p = self.net.res_bus.p_mw.values
        obs = np.concatenate((q, p, v))
        return obs

    def step(self, action):
        self.idx += 1
        try:
            self.adjust_net(action)
        except:
            info = 'sample idx out of bound'
            done = True
            v = self.net.res_bus.vm_pu.values
            q = self.net.res_bus.q_mvar.values
            p = self.net.res_bus.p_mw.values
            reward = self.calc_reward(v)
            next_obs = np.concatenate((q, p, v))
            return next_obs, reward, done, info

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
        self.net.sgen.q_mvar[list(self.gen_to_bus.keys())] += (action + 1) / 2 * (self.q_max - self.q_min) + self.q_min
        self.net.sgen.p_mw[list(self.gen_to_bus.keys())] = self.max_renewable_p_data[self.idx, list(self.gen_to_bus.keys())]
        self.net.load.p_mw[list(self.load_to_bus.keys())] = self.load_p_data[self.idx, list(self.load_to_bus.keys())]
        self.net.load.q_mvar[list(self.load_to_bus.keys())] = self.load_q_data[self.idx, list(self.load_to_bus.keys())]

    def calc_reward(self, voltages):
        if self.is_test:
            reward = -np.abs(voltages - 1).sum()
        else:
            reward = 2 - np.abs(voltages - 1).sum()
        return reward