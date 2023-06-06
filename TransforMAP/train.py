import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
#import sacrebleu
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
import numpy as np
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        #pem
        #torch.save(model.state_dict(), config.model_path)
    return total_loss / total_tokens


def train(train_data, dev_data, model, model_par, criterion, optimizer, model_save_path):
    """ Train and save model """
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # Model training
        model.train()
        # train_loss = run_epoch(train_data, model_par,
        #                       MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        train_loss = run_epoch(train_data, model_par,
                               LossCompute(model.generator, criterion, optimizer))

        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        torch.save(model.state_dict(), model_save_path)
        # Model validation
        if epoch%5==0:
            model.eval()
            dev_loss = run_epoch(dev_data, model_par,
                                 LossCompute(model.generator, criterion, None))
            bleu_score = evaluate(dev_data, model)
            #logging.info('Epoch: {}, Dev loss: {}'.format(epoch, dev_loss))
            logging.info('Epoch: {}, Dev loss: {}, Accuracy Score: {}'.format(epoch, dev_loss, bleu_score))

            # Update model if the bleu score is better on the current epoch
            if bleu_score > best_bleu_score:
                torch.save(model.state_dict(), model_save_path)
                best_bleu_score = bleu_score
                early_stop = config.early_stop
                logging.info("-------- Save Best Model! --------")
            else:
                early_stop -= 1
                logging.info("Early Stop Left: {}".format(early_stop))
            if early_stop == 0:
                logging.info("-------- Early Stop! --------")
                break


class LossCompute:
    """ Caculate loss and update back propagation parameters """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """ A multi-gpu loss compute and train function. """

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize


def evaluate(data, model, mode='dev', use_beam=False):
    """ Predict with trained model and print output """
    trg = []
    res = []
    with torch.no_grad():
        for batch in tqdm(data):
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            trg.extend(list(batch.trg.cpu().detach().numpy()))
            res.extend(decode_result)
    '''
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    '''
    #self.trg = trg[:, :-1]
    #bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    res_not_end=np.array(res)[:, :-1]
    trg_arr=np.array(trg)[:, 1:].reshape(-1)
    res_arr=np.array(res_not_end).reshape(-1)
    acc_score=accuracy_score(trg_arr,res_arr)
    return acc_score  #float(bleu.score)


def test(data, model, criterion,model_save_path):
    with torch.no_grad():
        # Load model
        model.load_state_dict(torch.load(model_save_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # Start prediction
        #test_loss = run_epoch(data, model_par,
        #                      MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        test_loss = run_epoch(train_data, model_par,
                               LossCompute(model.generator, criterion, None))

        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Accuracy Score: {}'.format(test_loss, bleu_score))


def translate(src, model, model_save_path,use_beam=False):
    """ Predict with trained model and print output """
    with torch.no_grad():
        #model = torch.nn.DataParallel(model)#Pem
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        return np.array(decode_result)[:,:-1]
        # no ending symbol
