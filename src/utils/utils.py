import os
import sys
import math
import json

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.hist = []
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.avg += (self.val - self.avg) * n / self.count

    def save_log(self):
        self.hist.append(self.avg)
        self.reset()


class TrainMonitor(object):
    """Record training"""

    def __init__(self, nb_iter=1, best_eval=math.inf, best_iter=1, early_stop=0):
        self.nb_iter = nb_iter
        self.best_eval = best_eval
        self.best_iter = best_iter
        self.early_stop = early_stop


    def state_dict(self):
        sd = {'nb_iter': self.nb_iter,
              'best_eval': self.best_eval,
              'best_iter': self.best_iter,
              'early_stop': self.early_stop,
            }
        return sd

    
    def load_state_dict(self, state_dict: dict):
        self.nb_iter = state_dict['nb_iter']
        self.best_eval = state_dict['best_eval']
        self.best_iter = state_dict['best_iter']
        self.early_stop = state_dict['early_stop']