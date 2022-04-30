from collections import namedtuple,deque
import random

translate = namedtuple('translate',
                        ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(translate(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)