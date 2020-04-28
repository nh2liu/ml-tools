from collections import deque
import numpy as np
import random
import torch

from ..util.npdata import NumpyCircularBuffer


class ExperienceReplayMemory:
    def __init__(self, mem_size = 1000):
        self.cache = deque(maxlen = mem_size)
    
    def add(self, state1, action, reward, state2, other):
        self.cache.append((state1, action, reward, state2, other))
    
    def sample(self, batch_size = 200):
        minibatch = random.sample(self.cache, batch_size)
        s1b = torch.cat([s1 for (s1, a, r, s2, other) in minibatch])
        s2b = torch.cat([s2 for (s1, a, r, s2, other) in minibatch])
        rb = torch.tensor([r for (s1, a, r, s2, other) in minibatch], dtype = torch.float32)
        ab = torch.tensor([a for (s1, a, r, s2, other) in minibatch], dtype = torch.long).reshape(-1, 1)
        otherb = [other for (s1, a, r, s2, other) in minibatch]
        
        ret = {
            'state1' : s1b,
            'actions': ab,
            'rewards': rb,
            'state2' : s2b,
            'other'  : otherb,
            'weights': torch.ones(len(rb))
        }
        return ret
    
    def update(self, errors):
        pass
    
    def __len__(self):
        return len(self.cache)


class PrioritizedExperienceReplayMemory:
    '''
    https://arxiv.org/pdf/1511.05952.pdf
    '''
    def __init__(self,
                 mem_size = 1000,
                 alpha = 0.6,
                 beta = 0.4,
                 epsilon = 0.0001,
                 beta_anneal_rate = 0.9995):
        self.cache = deque(maxlen = mem_size)
        self.priorities = NumpyCircularBuffer(mem_size)
        self.indexes = deque(maxlen = mem_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal_rate = beta_anneal_rate
        self.epsilon = epsilon
        
        self.index_cache = None
    
    def add(self, state1, action, reward, state2, other):
        experience = (state1, action, reward, state2, other)
        max_priority = self.priorities.max(default = 1)
        self.priorities.append(max_priority)
        self.cache.append(experience)
    
    def _priority_scale(self, alpha):
        c = np.power(self.priorities.unravel(), alpha)
        return c / sum(c)
    
    def _importance_sampling(self, probabilities, beta):
        annealed_beta = 1 - (1-beta) * self.beta_anneal_rate
        importance = np.power(probabilities * len(self.cache), -annealed_beta)
        return importance / np.max(importance)
    
    def update(self, errors, variant = 'direct'):
        if variant == 'direct':
            new_p = np.abs(errors) + self.epsilon
            self.priorities.batch_update(self.index_cache, new_p)
        elif variant == 'rank':
            # TODO: Rank update.
            pass
        else:
            raise Exception('Invalid variant for priority updates.')
    
    def sample(self, batch_size = 200):
        probabilities = self._priority_scale(self.alpha)
        weights = self._importance_sampling(probabilities, self.beta)
        n = len(self.cache)
        self.index_cache = np.random.choice(n, size = batch_size, replace = False, p = probabilities)
        minibatch = [self.cache[i] for i in self.index_cache]
        weights = torch.from_numpy(weights[self.index_cache])
        
        s1b = torch.cat([s1 for (s1, a, r, s2, other) in minibatch])
        s2b = torch.cat([s2 for (s1, a, r, s2, other) in minibatch])
        rb = torch.tensor([r for (s1, a, r, s2, other) in minibatch], dtype = torch.float32)
        ab = torch.tensor([a for (s1, a, r, s2, other) in minibatch], dtype = torch.long).reshape(-1, 1)
        otherb = [other for (s1, a, r, s2, other) in minibatch]
        
        ret = {
            'state1' : s1b,
            'actions': ab,
            'rewards': rb,
            'state2' : s2b,
            'other'  : otherb,
            'weights': weights
        }
        self.beta_anneal_rate *= self.beta_anneal_rate
        return ret
    
    def __len__(self):
        return len(self.cache)