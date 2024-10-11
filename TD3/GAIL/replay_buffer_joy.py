"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np


class ReplayBufferExp(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.count_exp = 0
        self.count_joy = 0
        self.count_exp_prev = 0
        self.count_badexp = 0
        self.buffer = deque()
        self.buffer_exp = deque()
        self.buffer_badexp = deque()
        self.buffer_headon = deque()
        self.buffer_stmove = deque()
        self.buffer_joy = deque()
        random.seed(random_seed)

    def add(self, s, a,  r, t, s2):
        experience = (s, a,  r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        # print("self.count_add", self.count)
    def add_exp(self, s, a, r, t, s2):
        experience_exp = (s, a, r, t, s2)
        if self.count_exp < self.buffer_size:
            self.buffer_exp.append(experience_exp)
            self.count_exp_prev = self.count_exp
            self.count_exp += 1
        else:
            self.buffer_exp.popleft()
            self.buffer_exp.append(experience_exp)

    def add_badexp(self, s, a, r, t, s2):
        experience_exp = (s, a, r, t, s2)
        if self.count_badexp < self.buffer_size:
            self.buffer_badexp.append(experience_exp)
            self.count_badexp += 1
        else:
            self.buffer_badexp.popleft()
            self.buffer_badexp.append(experience_exp)

    def add_joy(self, s, a, r, t, s2):
        experience_exp = (s, a, r, t, s2)
        if self.count_joy < self.buffer_size:
            self.buffer_joy.append(experience_exp)
            self.count_joy += 1
        else:
            self.buffer_exp.popleft()
            self.buffer_exp.append(experience_exp)

    def add_headon(self, headon):
        self.buffer_headon = headon


    def add_stmove(self, stmove):
        self.buffer_stmove = stmove


    def size(self):
        return self.count

    def sample_batch_exp(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer_exp, self.count)
        else:
            batch = random.sample(self.buffer_exp, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            if self.count_joy > 20:
                batch = random.sample(self.buffer, 20)
                batch_joy = random.sample(self.buffer_joy, 20)
                for i in batch_joy:
                    batch.append(i)
                np.random.shuffle(batch)
            else:
                batch = random.sample(self.buffer, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
    def sample_batch_badexp(self, batch_size):
        batch = []
        ran = random.randint(0, 9)
        if 0<=ran<3:
            if self.count < batch_size:
                batch = random.sample(self.buffer_badexp, self.count)
            else:
                batch = random.sample(self.buffer_badexp, batch_size)
        elif ran>=3:
            if self.count < batch_size:
                batch = random.sample(self.buffer, self.count)
            else:
                batch = random.sample(self.buffer, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
    def clear(self):
        self.buffer.clear()
        self.count = 0
