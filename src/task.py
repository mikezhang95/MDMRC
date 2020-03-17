"""


"""


import os
import sys
import time
import json
import logging
import numpy as np

from data_loader import get_data_loader
from utils import merge_dict

logger = logging.getLogger()


def train(model, train_loader, val_loader, config):

    # training parameters
    best_epoch_retriever, best_loss_retriever = 0, np.inf
    best_epoch_reader, best_loss_reader = 0, np.inf

    # models
    retriever, reader = model
    saved_models_retriever = []
    saved_models_reader = []
    last_n_model = config.last_n_model 

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('***** Training Begins at {} *****'.format(cur_time))
    for epoch in range(config.num_epoch):
        # EPOCH
        logger.info('\n***** Epoch {}/{} *****'.format(epoch, config.num_epoch))
        retriever.train()
        reader.train()

        num_batch = len(train_loader)
        for batch_cnt, batch in enumerate(train_loader):
            # BATCH
            loss_r1 =  retriever.retrieve(batch, mode="train")
            loss_r2 = reader.read(batch,mode="train")
            retriever.update(loss_r1)
            reader.update(loss_r2)

            # print training loss every print_frequency batch
            if (batch_cnt+1) % config.print_frequency == 0:
                logger.info("{}/{} Batch: Retriever-{:.4f}  Reader-{:.4f}".format(batch_cnt+1, num_batch, loss_r1, loss_r2))

        # Evaluate at the end of every epoch
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        logger.info('==== Evaluating Model at {} ===='.format(cur_time))

        # Validation (loss)
        loss_retriever, loss_reader = validate(model, val_loader)

        # Save Retriever
        if loss_retriever < best_loss_retriever:
            best_loss_retriever = loss_retriever 

            cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            logger.info('*** Retriever Saved with valid_loss = {}, at {}. ***'.format(loss_retriever, cur_time))
            retriever.save(config.saved_path, epoch)
            best_epoch_retriever = epoch
            saved_models_retriever.append(epoch)
            if len(saved_models_retriever) > last_n_model:
                remove_model = saved_models_retriever[0]
                saved_models_retriever = saved_models_retriever[-last_n_model:]
                os.remove(os.path.join(config.saved_path, "{}-retriever".format(remove_model)))

        # Save Reader
        if loss_reader < best_loss_reader:
            best_loss_reader = loss_reader 

            cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            logger.info('*** reader Saved with valid_loss = {}, at {}. ***'.format(loss_reader, cur_time))
            reader.save(config.saved_path, epoch)
            best_epoch_reader = epoch
            saved_models_reader.append(epoch)
            if len(saved_models_reader) > last_n_model:
                remove_model = saved_models_reader[0]
                saved_models_reader = saved_models_reader[-last_n_model:]
                os.remove(os.path.join(config.saved_path, "{}-reader".format(remove_model)))



    logger.info("Training Ends.\n Best Validation Loss: \
            - Retriever {:.4f} at Epcoh {}\n  \
            - Reader {:.4f} at Epoch {}".format(best_loss_retriever, best_epoch_retriever,best_loss_reader, best_epoch_reader))
 
    return best_epoch_retriever, best_epoch_reader


def validate(model, data_loader):
    # models
    retriever, reader = model
    retriever.eval()
    reader.eval()

    # validate result
    loss_retriever, loss_reader = 0.0, 0.0
    metric_retriever, metric_reader = {}, {}

    for i, batch in enumerate(data_loader):

        # forward pass
        loss1 =  retriever.retrieve(batch, mode="train")
        loss_retriever += loss1
        loss2 = reader.read(batch,mode="train")
        loss_reader += loss2

        # calculate topk for retriever
        metric1 = retriever.collect_metric(batch)
        merge_dict(metric_retriever, metric1)

        # calculate bleu/f1/rouge
        metric2 = reader.collect_metric(batch)
        merge_dict(metric_reader, metric2)


    # TODO: save badcase

    num_batch = len(data_loader)
    loss_retriever /= float(num_batch)
    loss_reader /= float(num_batch)

    # YZ: calculate mean right now. Analyss badcase later
    for k,v in metric_retriever.items():
        metric_retriever[k] = np.mean(v)

    logger.info('\n--- Retriever loss = {}'.format(loss_retriever))
    logger.info(json.dumps(metric_retriever))

    logger.info('\n--- Reader loss = {}'.format(loss_reader))
    logger.info(json.dumps(metric_reader))
    sys.stdout.flush()
    return loss_retriever, loss_reader


def generate(model, data_loader):
    # models
    retriever, reader = model
    retrieve.eval()
    # reader.eval()

    for batch in data_loader:
        retriever.predict(batch) # generate doc candidates
        answer = reader.predict(batch) # generate answer

    write_result()
    return


# def pre_train():
    # # mask words to pretrain language model
    # pass

# def joint_train():
    # # joint train retriever and reader
#     pass



