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
import math
import logging
import argparse
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils import *
from data_loader import load_data, get_data_loader
from task import  train, validate, generate

import retrievers
import readers


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default="bm25_bert")
parser.add_argument('--forward_only', action='store_true')
parser.add_argument('--alias', type=str, default="")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--localhost',type=str,default="23456")
parser.add_argument('--add_gp',action='store_true')
parser.add_argument('--gp_epsilon',type=float,default=0.4)
parser.add_argument('--add_noise_labels',action='store_true')
parser.add_argument('--noise_labels_offset_bound',type=int,default=1)#s,e offset
parser.add_argument('--add_typeloss',action='store_true')
parser.add_argument('--typeloss_epsilon',type=float,default=1)
parser.add_argument('--val_epoch_ratio',type=float,default=1)
args = parser.parse_args()


# load config
config_path = BASE_DIR + "configs/" + args.config_name + ".conf"
config = Pack(json.load(open(config_path)))
config["forward_only"] = args.forward_only
config["debug"] = args.debug
config["add_gp"]=args.add_gp
config["add_noise_labels"]=args.add_noise_labels
config["add_typeloss"]=args.add_typeloss
config["val_epoch_ratio"]=args.val_epoch_ratio
if args.add_gp:
    config["gp_epsilon"]=args.gp_epsilon
if args.add_noise_labels:
    config["noise_labels_offset_bound"]=args.noise_labels_offset_bound
if args.add_typeloss:
    config["typeloss_epsilon"]=args.typeloss_epsilon
    
# set gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
config.use_gpu = torch.cuda.device_count() > 0
if config.use_gpu:
    torch.distributed.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.localhost}', rank=0, world_size=1)

# set random_seed/logger/save_path
set_seed(config.random_seed)
stats_path = BASE_DIR + 'outputs/'
alias = args.alias if args.alias == "" else '-'+ args.alias
saved_path = os.path.join(stats_path, args.config_name + alias)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
config.saved_path = saved_path
prepare_dirs_loggers(config)

# start logger
logger = logging.getLogger()
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))

# load dataset 
train_data, test_data, documents = load_data(config)
# split dataset into train/val 4:1
config.batch_size = math.floor(config.batch_size*1.0/config.gradient_accumulation_steps)
train_loader, val_loader = get_data_loader(train_data, batch_size=config.batch_size, split_ratio=0.2, use_gpu=config.use_gpu, shuffle=True)
test_loader,_ = get_data_loader(test_data, batch_size=config.test_batch_size, split_ratio=0.0, use_gpu=config.use_gpu, shuffle=False)
config["num_samples"]= len(train_data)


# create model
retriever_class = getattr(retrievers, config.retriever_name)
retriever = retriever_class(documents, config)
reader_class = getattr(readers, config.reader_name)
reader = reader_class(documents, config)
model = (retriever, reader)


# load pretrain model before training
if not config.forward_only and config.pretrain_folder != "":
    pretrain_path = os.path.join(stats_path, config.pretrain_folder)
    logger.info(f"use pretrain in dir {pretrain_path}")
    best_epoch = find_best_model(pretrain_path)
    logger.info(f"[retriever,reader]={best_epoch}")
    retriever.load(pretrain_path, best_epoch[0])
    reader.load(pretrain_path, best_epoch[1])

##################### Training #####################
"""
NOTE: best_epoch不再代表最好效果的那一个epoch，而代表最好效果的那个step后那个point（考虑到grad_acc，所以一个update_batch一个step）
结合best_epoch,num_batch,grad_acc可以知道所存模型处于哪个epoch的第几个batch的阶段保存的
"""
best_epoch = None
if not config.forward_only:
    # save config
    with open(os.path.join(saved_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)  # sort_keys=True
    try:
        best_epoch = train(model, train_loader, val_loader, config)
    except KeyboardInterrupt:
        logger.error('Training stopped by keyboard.')

# load best model
best_epoch = find_best_model(saved_path)
retriever.load(saved_path, best_epoch[0])
reader.load(saved_path, best_epoch[1])
logger.info("$$$ Load {}-model $$$".format(best_epoch))

best_epoch = "%s-%s"%(best_epoch[0], best_epoch[1])
##################### Validation #####################
logger.info("\n***** Evaluation on VAL *****")
with open(os.path.join(saved_path, '{}-badcase.csv'.format(best_epoch)), 'w') as f:
    validate(model, val_loader, f)

##################### Generation #####################
logger.info("\n***** Generation on TEST *****")
# f2 = open(os.path.join(saved_path, 'test_logits.csv'),"w")
with open(os.path.join(saved_path, '{}-prediction.csv'.format(best_epoch)), 'w') as f:
    # generate(model, test_loader, pred_f=f, logit_f=f2)
    generate(model, test_loader, pred_f=f)

end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[END]\n' + end_time)
