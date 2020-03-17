
import os
import sys
import logging
import random
import time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../'
DATA_DIR =  BASE_DIR + 'data/'

logger = logging.getLogger()

def merge_dict(dict_old, dict_new):
    for k,v in dict_new.items():
        if k in dict_old:
            dict_old[k].extend(v)
        else:
            dict_old[k] = v


def to_numpy(tensor):
    return tensor.cpu().numpy()

def to_torch(array, use_gpu=False, dtype=torch.long):
    tensor = torch.from_numpy(array).to(dtype)
    if use_gpu:
        tensor = tensor.cuda()
    else:
        use_gpu = tensor.gpu()
    return tensor

def pad_sequence(seqs, pad=None, max_len=None):
    if not max_len:       
        max_len = max([len(s) for s in seqs])
    if not pad:
        pad = 0
    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            seqs[i] = seqs[i][:max_len]
        else:
            seqs[i].extend([pad] * (max_len - len(seqs[i])))
    return seqs


class Pack(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return False

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

# def cast_type(var, dtype, use_gpu):
    # if use_gpu:
        # if dtype == INT:
            # var = var.type(th.cuda.IntTensor)
        # elif dtype == LONG:
            # var = var.type(th.cuda.LongTensor)
        # elif dtype == FLOAT:
            # var = var.type(th.cuda.FloatTensor)
        # else:
            # raise ValueError('Unknown dtype')
    # else:
        # if dtype == INT:
            # var = var.type(th.IntTensor)
        # elif dtype == LONG:
            # var = var.type(th.LongTensor)
        # elif dtype == FLOAT:
            # var = var.type(th.FloatTensor)
        # else:
            # raise ValueError('Unknown dtype')
    # return var


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_dirs_loggers(config, script=""):

    log_level = logging.INFO

    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(log_level)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(log_level)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if hasattr(config, 'forward_only') and config.forward_only:
        return

    fileHandler = logging.FileHandler(os.path.join(config.saved_path,'session.log'))
    fileHandler.setLevel(log_level)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    return


import functools
def cost(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        ret = func(*args, **kw)
        end_time = time.time()
        logger.debug('@%s took %.6f seconds' % (func.__name__, end_time-start_time))
        return ret
    return wrapper

