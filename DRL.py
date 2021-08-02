from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import time


class ANet(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(ANet, self).__init__()
        self.a_bound = a_bound
        self.fc1 = nn.Linear(s_dim, 64)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # new
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(64, 128)

        self.out1 = nn.Linear(128, int(a_dim / 2))
        self.out2 = nn.Linear(128, int(a_dim / 2))
        # self.out1.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x2 = F.relu(self.fc3(x))
        x1 = torch.sigmoid((self.out1(x1)))
        # print(x1)
        # x2=F.sigmoid(self.out2(x2))
        x2 = torch.softmax(self.out2(x2), -1)
        actions_value = torch.cat((x1, x2), dim=-1)
        # actions_value = actions_value * self.a_bound.item()
        return actions_value


class CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        # self.fcs.weight.data.normal_(0, 0.1)  # initialization

        self.fca = nn.Linear(a_dim, 30)
        # self.fca.weight.data.normal_(0, 0.1)  # initialization

        self.out = nn.Linear(30, 1)
        # self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        actions_value = self.out(net)  # V(s,a)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.pointer = 0  # exp buffer
        self.lr_a = 1e-6
        # learning rate for actor
        self.lr_c = 1e-2  # learning rate for critic
        self.gamma = 0.999  # reward discount
        self.tau = 1e-3  # soft update
        self.memory_capacity = 100000
        self.batch_size = 128
        self.time = time.time()

        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        # self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(self.device)

        self.Actor_eval = ANet(s_dim, a_dim, a_bound).to(self.device)  # main
        self.Actor_target = ANet(s_dim, a_dim, a_bound).to(self.device)  # target
        self.Critic_eval = CNet(s_dim, a_dim).to(self.device)  # main
        self.Critic_target = CNet(s_dim, a_dim).to(self.device)  # taget
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=self.lr_c)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=self.lr_a)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s).to(self.device), 0)
        return self.Actor_eval(s)[0].detach().cpu()

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1 - self.tau))')
            eval('self.Actor_target.' + x + '.data.add_(self.tau * self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1- self.tau))')
            eval('self.Critic_target.' + x + '.data.add_(self.tau * self.Critic_eval.' + x + '.data)')

        # soft target replacement

        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(self.device)  # state
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(self.device)  # action
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(self.device)  # reward
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(self.device)  # next state

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce(s,ae(s))   ae(s)=a   ae(s_)=a_

        loss_a = -torch.mean(q)

        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)
        q_ = self.Critic_target(bs_, a_)
        q_target = br + self.gamma * q_

        q_v = self.Critic_eval(bs, ba)

        td_error = self.loss_td(q_target, q_v)

        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_mode(self):
        torch.save(self.Actor_eval, "./checkpoint/best_model.pth")

    def load_model(self, name=None):
        if name:
            self.time = name
        self.Actor_eval = torch.load("./checkpoint/best_model.pth")
        self.Actor_eval.to(self.device)
