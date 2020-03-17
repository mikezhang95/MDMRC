"""
    This is the main entry of multiwoz experiments(supervised learning)
    Usage:
        - train and evaluate
            python supervised.py --config_name sl_cat
        - evaluate only
            python supervsied.py --config_name sl_cat --forward_only
"""

import os
import sys
import time
import json
import logging
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import Pack, prepare_dirs_loggers, set_seed
from data_loader import load_data, get_data_loader
from task import  train, validate

import retrievers
import readers



# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default="bm25_bert")
parser.add_argument('--forward_only', action='store_true')
parser.add_argument('--alias', type=str, default="")
args = parser.parse_args()


# load config
config_path = BASE_DIR + "configs/" + args.config_name + ".conf"
config = Pack(json.load(open(config_path)))
config["forward_only"] = args.forward_only


# set random_seed/logger/save_path
set_seed(config.random_seed)
stats_path = BASE_DIR + 'outputs/'
if config.forward_only:
    saved_path = os.path.join(stats_path, config.pretrain_folder)
    config = Pack(json.load(open(os.path.join(saved_path, 'config.json'))))
    config['forward_only'] = True
else:
    alias = args.alias
    if alias != "" :
        alias = '-' + alias
    saved_path = os.path.join(stats_path, args.config_name + alias)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
config.saved_path = saved_path
prepare_dirs_loggers(config)

# start logger
logger = logging.getLogger()
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))


# save config
with open(os.path.join(saved_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)  # sort_keys=True


# load dataset 
train_data, test_data, documents = load_data(config)
# split dataset into train/val 4:1
train_loader, val_loader = get_data_loader(train_data, batch_size=config.batch_size, split_ratio=0.2)
config["num_samples"]= len(train_data)

# create model
retriever_class = getattr(retrievers, config.retriever_name)
retriever = retriever_class(documents, config)
reader_class = getattr(readers, config.reader_name)
reader = reader_class(documents, config)
if config.use_gpu:
    retriever = retriever.cuda()
    reader = reader.cuda()
model = (retriever, reader)

##################### Training #####################
best_epoch = None
if not config.forward_only:
    try:
        best_epoch = train(model, train_loader, val_loader, config)
    except KeyboardInterrupt:
        logger.error('Training stopped by keyboard.')
if best_epoch is None:
    retriever_models = sorted([int(p.replace('-retriever', '')) for p in os.listdir(saved_path) if 'retriever' in p])
    reader_models = sorted([int(p.replace('-reader', '')) for p in os.listdir(saved_path) if 'reader' in p])
    # best_epoch = (retriever_models[-1], reader_modes[-1])
    best_epoch = (retriever_models[-1], reader_models[-1])

# load best model
retriever.load(saved_path, best_epoch[0])
reader.load(saved_path, best_epoch[1])


##################### Validation #####################
logger.info("\n***** Evaluation on VAL *****")
logger.info("$$$ Load {}-model $$$".format(best_epoch))
validate(model, val_loader)

# ##################### Generation #####################
# # TODO: support write into files

# with open(os.path.join(saved_path, '{}_test_file.txt'.format(best_epoch)), 'w') as f:
    # generate(model, test_data, config, evaluator, dest_f=f)

end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[END]\n' + end_time)
