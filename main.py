import argparse
import logging
import time

import numpy as np
import torch
from env import FL_env
from load_config import Config
from DRL import DDPG


def lim_sum(array):
    return array / sum(array)


def act_limit(a, dim):
    f = a[0:int(dim / 2)]
    f = np.clip(f, 0.1, 1)
    w = lim_sum(a[int(dim / 2):])
    w = np.clip(w, 0.01, 1)
    return np.concatenate((f, w))


def test(env, ddpg, times):
    ddpg.load_model("./checkpoint/best_model.pth")
    res = {"ddpg": [0, 0, 0], "static": [0, 0, 0], "bandwith": [0, 0, 0], "EF": [0, 0, 0]}
    for i in range(times):
        for obj in ["ddpg", "static", "bandwith", "EF"]:
            s = env.reset()
            step = 0
            ep_reward = 0
            total_time = 0
            total_energy = 0
            done = False
            while not done:
                step += 1
                if obj == "static":
                    a = np.array([0.3] * config.clients.num + [1 / config.clients.num] * config.clients.num)
                else:
                    a = ddpg.choose_action(s)
                    # print(torch.sum(a[config.clients.num:]))

                if obj == "bandwith":
                    a[config.clients.num:] = torch.tensor([1 / config.clients.num] * config.clients.num)
                if obj == "EF":
                    a[config.clients.num:] = torch.tensor(env.EF())
                a = act_limit(a, env.act_dim)
                next_obs, reward, done, latency, energy = env.step(a)
                s = next_obs
                # print(s1)
                ep_reward += reward
                total_time += latency
                total_energy += energy
                if done:
                    res[obj][0] += step
                    # print(total_time)
                    res[obj][1] += total_time
                    res[obj][2] += total_energy

    for obj in ["ddpg", "static", "bandwith", "EF"]:
        print("---------------{}------------".format(obj))
        step = res[obj][0] / times
        latency = res[obj][1] / times / step
        energy = res[obj][2] / times / step
        print("step: {}, latency: {}, energy: {}".format(step, latency, energy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.json',
                        help='configuration file.')
    parser.add_argument('-l', '--log', type=str, default='INFO',
                        help='Log messages level.')
    parser.add_argument('-r', '--render', type=bool, default=False,
                        help='render')
    args = parser.parse_args()
    # Set logging
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()),
        datefmt='%H:%M:%S')
    logging.info("config:{},  log:{}".format(args.config, args.log))
    # load config
    config = Config(args.config)
    # load environment
    env = FL_env(config)
    # set DRL agent
    s_dim = env.obs_dim  # state space
    a_dim = env.act_dim  # action space
    a_bound = 1  # action bound
    ddpg = DDPG(a_dim, s_dim, a_bound)
    var = 0.8  # variance of exploration
    var2 = 0.5
    t1 = time.time()
    MAX_EP_STEPS = 1000
    rewards = []
    times = []
    en = []
    stp = []
    start_step = 5
    best_reward = 1

    for i in range(800):
        # reset state
        s = env.reset()
        ep_reward = 0
        total_time = 0
        total_energy = 0
        for j in range(MAX_EP_STEPS):
            a = ddpg.choose_action(s)
            # add randomness to action selection for exploration
            a[:config.clients.num] += 0.1 * np.random.normal(0, var, config.clients.num)
            a[config.clients.num:] += 0.01 * np.random.normal(0, var2, config.clients.num)
            a = act_limit(a, a_dim)
            # take action
            next_obs, reward, done, latency, energy = env.step(a)
            ddpg.store_transition(s, a, reward, next_obs)
            # update next state
            s = next_obs
            ep_reward += reward
            total_time += latency
            total_energy += energy

            # When the game is over
            if j == MAX_EP_STEPS - 1 or done == True:
                if i > 20: # DDPG does not update in the first 20 epochs
                    # update 50 times
                    for t in range(0, 50):
                        ddpg.learn()
                if ep_reward > best_reward:
                    ddpg.save_mode()
                    best_reward = ep_reward
                print(
                    "Episode {}: max_step: {} reward {:2f}, latency: {:.2f}, energy: {:.2f}  best_reward: {:.2f}".format(i, j,
                                                                                                                   ep_reward,
                                                                                                                   total_time / j,
                                                                                                                   total_energy / j,

                                                                                                                   best_reward))
                rewards.append(ep_reward)
                times.append(total_time / j)
                en.append(total_energy / j)
                stp.append(j)
                # decay the action randomness
                var *= .999
                var2 *= .999
                break
    print('Running time: ', time.time() - t1)
    test(env, ddpg, 20)
