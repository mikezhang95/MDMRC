"""


"""


import os
import sys
import logging

logger = logging.getLogger()

def train(model, train_data, val_data, config, evaluator):
    tb_path = os.path.join(config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    # training parameters
    patience, batch_cnt, best_epoch = 10, 0, 0
    valid_loss_threshold, best_valid_loss = np.inf, np.inf
    train_loss = LossManager()

    # models
    model.train()
    optimizer = model.get_optimizer(config, verbose=False)
    saved_models = []
    last_n_model = config.last_n_model 


    logger.info('***** Training Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info('***** Epoch 0/{} *****'.format(config.num_epoch))
    for epoch in range(config.num_epoch):
        # EPOCH
        train_data.epoch_init(config, shuffle=True, verbose=epoch==0, fix_batch=config.fix_train_batch)
        num_batch = train_data.num_batch

        while True:
            # BATCH
            batch = train_data.next_batch()
            if batch is None:
                break

            optimizer.zero_grad()
            # TODO: TEACH_FORCE = superviser learning ?
            loss = model(batch, mode=TEACH_FORCE)
            train_loss.add_loss(loss) 
            model.backward(loss, batch_cnt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_cnt += 1

            # tensorboard save 
            data_dict = {}
            for key, val in loss.items():
                if val is not None and type(val) is not bool:
                    data_dict["train/%s"%key] = val.item()
            tb_logger.add_scalar_summary(data_dict, batch_cnt)

            # print training loss every print_frequency batch
            if batch_cnt % config.print_frequency == 0:
                # TODO: what is model.kl_w
                logger.info(train_loss.pprint('Train',
                                        window=config.print_frequency,
                                        prefix='{}/{}-({:.3f})'.format(batch_cnt%num_batch, num_batch, model.kl_w)))
                sys.stdout.flush()

        # Evaluate at the end of every epoch
        logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        logger.info('==== Evaluating Model ====')

        # Generation (bleu/success/match)
        success, match, bleu = generate(model, val_data, config, evaluator)

        # Validation (loss)
        logger.info(train_loss.pprint('Train'))
        valid_loss = validate(model, val_data, config, batch_cnt)

        stats = {'val/success': success, 'val/match': match, 'val/bleu': bleu, "val/loss": valid_loss}
        tb_logger.add_scalar_summary(stats, batch_cnt)

        # Save Models if valid loss decreases
        if valid_loss < best_valid_loss:
            if valid_loss <= valid_loss_threshold * config.improve_threshold:
                patience = max(patience, epoch*config.patient_increase)
                valid_loss_threshold = valid_loss
                logger.info('Update patience to {}'.format(patience))

            if config.save_model:
                cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                logger.info('*** Model Saved with valid_loss = {}, at {}. ***'.format(valid_loss, cur_time))
                model.save(config.saved_path, epoch)
                best_epoch = epoch
                saved_models.append(epoch)
                if len(saved_models) > last_n_model:
                    remove_model = saved_models[0]
                    saved_models = saved_models[-last_n_model:]
                    os.remove(os.path.join(config.saved_path, "{}-model".format(remove_model)))

            best_valid_loss = valid_loss

        # Early stop 
        # By: YZ. currently no early stopping in configs
        if config.early_stop and patience <= epoch:
            logger.info('*** Early stop due to run out of patience ***')
            break

        # exit val mode
        model.train()
        train_loss.clear()
        logger.info('\n***** Epoch {}/{} *****'.format(epoch+1, config.num_epoch))
        sys.stdout.flush()
    
    logger.info('Training Ends. Best validation loss = %f' % (best_valid_loss, ))
    return best_epoch

def validate(model, data, config, batch_cnt=None):
    model.eval()
    data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)
    losses = LossManager()
    while True:
        batch = data.next_batch()
        if batch is None:
            break
        loss = model(batch, mode=TEACH_FORCE)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()
    logger.info(losses.pprint(data.name))
    logger.info('--- Total loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss



def pre_train():
    # mask words to pretrain language model
    pass

def joint_train():
    # joint train retriever and reader
    pass
