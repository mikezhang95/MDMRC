"""


"""


import os
import sys
import time
import logging
import numpy as np

from data_loader import get_data_loader

logger = logging.getLogger()


def train(model, train_data, config):

    # training parameters
    best_epoch, best_valid_loss = 0, np.inf

    # models
    retriever, reader = model
    retriever.train()
    reader.train()
    saved_models = []
    last_n_model = config.last_n_model 

    # split dataset into train/val 4:1
    train_loader, val_loader = get_data_loader(train_data, batch_size=config.batch_size, split_ratio=0.2)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('***** Training Begins at {} *****'.format(cur_time))
    for epoch in range(config.num_epoch):
        # EPOCH
        logger.info('\n***** Epoch {}/{} *****'.format(epoch, config.num_epoch))

        for batch_cnt, batch in enumerate(train_loader):
            # BATCH

            # optimizer.zero_grad()
            loss_r1 =  retriever(batch, topk=1)
            loss_r2 = reader(batch)

            retriever.update(loss_r1)
            reader.update(loss_r2)
            # optimizer.step()


            # print training loss every print_frequency batch
            if (batch_cnt+1) % config.print_frequency == 0:
                pass

        # Evaluate at the end of every epoch
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        logger.info('==== Evaluating Model at {} ===='.format(cur_time))


        # Validation (loss)


#         # Save Models if valid loss decreases
        # if valid_loss < best_valid_loss:
            # if valid_loss <= valid_loss_threshold * config.improve_threshold:
                # patience = max(patience, epoch*config.patient_increase)
                # valid_loss_threshold = valid_loss
                # logger.info('Update patience to {}'.format(patience))

            # if config.save_model:
                # cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                # logger.info('*** Model Saved with valid_loss = {}, at {}. ***'.format(valid_loss, cur_time))
                # model.save(config.saved_path, epoch)
                # best_epoch = epoch
                # saved_models.append(epoch)
                # if len(saved_models) > last_n_model:
                    # remove_model = saved_models[0]
                    # saved_models = saved_models[-last_n_model:]
                    # os.remove(os.path.join(config.saved_path, "{}-model".format(remove_model)))
            # best_valid_loss = valid_loss

    # retriever.train()
    # reader.train()

    # logger.info('Training Ends. Best validation loss = %f' % (best_valid_loss, ))
    # return best_epoch


# # def validate(model, data, config, batch_cnt=None):
    # # model.eval()
    # # data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)
    # # losses = LossManager()
    # while True:
        # batch = data.next_batch()
        # if batch is None:
            # break
        # loss = model(batch, mode=TEACH_FORCE)
        # losses.add_loss(loss)
        # losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    # valid_loss = losses.avg_loss()
    # logger.info(losses.pprint(data.name))
    # logger.info('--- Total loss = {}'.format(valid_loss))
    # sys.stdout.flush()
    # return valid_loss



# def pre_train():
    # # mask words to pretrain language model
    # pass

# def joint_train():
    # # joint train retriever and reader
#     pass

# def kfold_train():
    # pass



