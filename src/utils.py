
CUR_DIR = os.path.join(os.path.dirname(__file__)) + '/'
DATA_DIR = CUR_DIR + "../data/"

# class Pack(dict):
    # def __getattr__(self, name):
        # if name in self:
            # return self[name]
        # else:
            # return False

    # def add(self, **kwargs):
        # for k, v in kwargs.items():
            # self[k] = v

    # def copy(self):
        # pack = Pack()
        # for k, v in self.items():
            # if type(v) is list:
                # pack[k] = list(v)
            # else:
                # pack[k] = v
        # return pack

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

# def read_lines(file_name):
    # """Reads all the lines from the file."""
    # assert os.path.exists(file_name), 'file does not exists %s' % file_name
    # lines = []
    # with open(file_name, 'r') as f:
        # for line in f:
            # lines.append(line.strip())
    # return lines

# def set_seed(seed):
    # """Sets random seed everywhere."""
    # th.manual_seed(seed)
    # if th.cuda.is_available():
        # th.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

# def prepare_dirs_loggers(config, script=""):
    # logFormatter = logging.Formatter("%(message)s")
    # rootLogger = logging.getLogger()
    # rootLogger.setLevel(logging.DEBUG)

    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setLevel(logging.DEBUG)
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)

    # if hasattr(config, 'forward_only') and config.forward_only:
        # return

    # fileHandler = logging.FileHandler(os.path.join(config.saved_path,'session.log'))
    # fileHandler.setLevel(logging.DEBUG)
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)


