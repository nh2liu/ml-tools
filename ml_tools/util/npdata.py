import numpy as np

class NumpyCircularBuffer:
    '''
    Circular buffer that uses a numpy array as the cache.
    Probably could be optimized but implementation works pretty well
    for current uses.
    '''
    def __init__(self, max_len, shape = None):
        self.max_len = max_len
        self.cur_len = 0
        if shape == None:
            self._buffer = np.zeros(max_len)
        else:
            self._buffer = np.zeros((max_len, *shape))
        self._idx = 0
    
    def __len__(self):
        return self.cur_len
    
    def append(self, x):
        self._buffer[self._idx] = x
        if self.cur_len < self.max_len:
            self.cur_len += 1
        self._idx = (self._idx + 1) % self.max_len
    
    def unravel(self):
        if self.cur_len < self.max_len:
            return self._buffer[:self.cur_len]
        return np.concatenate((self._buffer[self._idx:], self._buffer[:self._idx]))
    
    def __getitem__(self, i):
        return self._buffer[(self._idx + i) % self.max_len]
    
    def __setitem__(self, i, v):
        self._buffer[(self._idx + i) % self.max_len] = v
        return v
    
    def __repr__(self):
        return self.unravel().__repr__()
    
    def max(self, default = 0):
        if self.cur_len == 0: return default
        return np.max(self._buffer)
    
    def batch_update(self, indexes, values):
        converted_indexes = np.mod(indexes + self._idx, self.max_len)
        self._buffer[converted_indexes] = values
    