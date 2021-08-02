import numpy as np



def softmax(array):
    e = np.exp(array)
    return e / np.sum(e)


class FL_env:
    def __init__(self, config):
        # initialization
        self.config = config
        self._max_episode_steps = 1000
        self.lambada = self.config.DRL.lambada
        self.h = self.config.wireless.h

        self.user_battery = np.random.uniform(self.config.clients.battery_min, self.config.clients.battery_max,
                                              self.config.clients.num)
        self.battery = self.user_battery
        self.trans_p = np.random.uniform(self.config.clients.p_min, self.config.clients.p_max,
                                         self.config.clients.num)
        self.cycles = np.random.uniform(self.config.clients.cycles_min, self.config.clients.cycles_max,
                                        self.config.clients.num)
        self.coefficient = np.random.uniform(self.config.clients.coefficient_min, self.config.clients.coefficient_max,
                                             self.config.clients.num)
        self.data_size = np.random.uniform(self.config.clients.data_size_min, self.config.clients.data_size_max,
                                           self.config.clients.num)

        self.lambada = self.config.DRL.lambada

        self.work = np.array([1] * self.config.clients.num)

        self.f_max = np.random.uniform(self.config.clients.f_min, self.config.clients.f_max, self.config.clients.num)
        self.w_max = np.random.uniform(self.config.wireless.bandwidth_min, self.config.wireless.bandwidth_max)
        self.h2_init = np.random.exponential(self.h, self.config.clients.num)
        # state
        self.state = self.reset()

        # dimension of obs/act space
        self.obs_dim = int(np.hstack(self.state).shape[0])
        self.act_dim = self.config.clients.num * 2
        self.work = np.array([1] * self.config.clients.num)

        self.parm = 0

    def seed(self, n):
        np.random.seed(n)

    def reset(self):
        self.user_battery = self.battery
        # self.h2 = np.random.exponential(self.h, self.config.clients.num)
        # w_max = np.random.uniform(self.config.wireless.bandwidth_min, self.config.wireless.bandwidth_max)

        self.h2 = self.h2_init
        obs = np.hstack([np.array([0]) / self.config.FL.epoch, self.user_battery / self.config.clients.battery_max,
                         self.f_max / self.config.clients.f_max, self.h2 / self.h2_init,
                         self.w_max / self.config.wireless.bandwidth_max])
        self.state = obs
        self.is_end = False
        self.count = 0

        return self.state

    def step(self, action):
        epoch, battery, f_max, w_max = self.state[0] * self.config.FL.epoch, self.state[
                                                                             1:1 + self.config.clients.num] * self.config.clients.battery_max, self.state[
                                                                                                                                               1 + self.config.clients.num:1 + 2 * self.config.clients.num] * self.config.clients.f_max, \
                                       self.state[-1] * self.config.wireless.bandwidth_max

        f, w = action[0:self.config.clients.num], action[self.config.clients.num:2 * self.config.clients.num]
        bandwidth = w_max * w

        # frequency
        frequency = f_max * f

        #
        latency, transmission_time = self.computing_time(frequency, bandwidth)
        energy = self.computing_energy(frequency, transmission_time)
        total_energy = np.sum(energy)
        time = max(latency)
        cost = self.lambada * time + (1 - self.lambada) * total_energy
        # cost = time
        reward = epoch / 100 - 0.01 * cost
        # update state
        # simplified operation
        battery = np.maximum(battery - energy, 0)
        count = 0

        for index in range(0, len(battery)):
            if battery[index] == 0:
                count += 1
                # self.work[index]=0

        if count > 0:
            # reward=-1000/epoch
            self.is_end = True

        if epoch == self.config.FL.epoch - 1:
            # reward=100
            self.is_end = True


        r1 = np.random.uniform(0.8, 1.2, self.config.clients.num)

        f_max = self.f_max * r1
        # w_max = np.random.uniform(self.config.wireless.bandwidth_min, self.config.wireless.bandwidth_max)
        r2 = np.random.uniform(0.8, 1.2, self.config.clients.num)
        self.h2 = self.h2_init * r2
        epoch = epoch + 1
        next_state = np.hstack([np.array([epoch]) / self.config.FL.epoch, battery / self.config.clients.battery_max,
                                f_max / self.config.clients.f_max, self.h2 / self.h2_init,
                                w_max / self.config.wireless.bandwidth_max])
        self.state = next_state

        return next_state, reward, self.is_end, time, total_energy

    def computing_time(self, frequency, bandwidth):
        # latency of training
        t1 = self.work * (self.config.FL.E * self.cycles * self.data_size) / frequency
        # latency of uploading
        SNR = (self.trans_p * self.h2) / self.config.wireless.variance
        rate = bandwidth * np.log2(1 + SNR)
        t2 = self.work * 83886080 / rate
        latency = t1 + t2

        return latency, t2

    def computing_energy(self, frequency, transmission_time):
        # energy consumption of training
        e1 = self.coefficient * frequency * frequency * self.cycles * self.data_size * self.config.FL.E
        e2 = transmission_time * self.trans_p
        return (e1 + e2)

    def action_space_sample(self):
        f = np.array([0.4] * self.config.clients.num) + 0.1 * np.random.normal(0, 0.2, self.config.clients.num)
        w = np.array([0.5] * self.config.clients.num) + 0.01 * np.random.normal(0, 0.3, self.config.clients.num)
        return np.hstack([f, w])

    # energy first
    def EF(self):
        SNR = (self.trans_p * self.h2) / self.config.wireless.variance
        shannon = np.log2(1 + SNR)
        k = self.trans_p * 83886080 / shannon
        # print(np.sqrt(k)/np.sum(np.sqrt(k)))
        return np.sqrt(k) / np.sum(np.sqrt(k))
