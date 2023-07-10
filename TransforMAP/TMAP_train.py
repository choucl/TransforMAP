import utils
import config
import logging
import json
import gzip
import os
import numpy as np
import pandas as pd
import torch
from torchinfo import summary
from torch.utils.data import DataLoader

from train import train, translate
from data_loader import MAPDataset
from model import make_model, LabelSmoothing

# %% Preprocessing
BLOCK_BITS = config.BLOCK_BITS
TOTAL_BITS = config.TOTAL_BITS
VOCAB_SIZE = config.VOCAB_SIZE
LOOK_BACK = config.LOOK_BACK
SKIP_FORWARD = config.SKIP_FORWARD
PRED_FORWARD = config.PRED_FORWARD

PAD_ID = config.PAD_ID
START_ID = config.START_ID
END_ID = config.END_ID
BLOCK_NUM_BITS = config.BLOCK_NUM_BITS
PAGE_BITS = config.PAGE_BITS
BITMAP_SIZE = config.BITMAP_SIZE
OFFSET = config.OFFSET
# %%% Interface


def read_load_trace_data(json_path, trace_dir, work_group,
                         train_split, sp_stat=None):

    def get_train_file_name():
        j = open(json_path)
        db = json.load(j)
        training_files = []  # file names for training simpoints
        work_group_list = work_group.split(",")
        spt_count = {}
        for item in db["group_items"]:
            # select work groups
            if (work_group == 'all' or item in work_group_list):
                spt_count[item] = 0
                for sub_item, num_spts in db["num_spts"][item].items():
                    spt_count[item] += num_spts
                    if (sp_stat is None):
                        for i in range(1, num_spts + 1):
                            result_dir = trace_dir + \
                                 "/%s/%s/simpoint_%d" % (item, sub_item, i)
                            result_file = result_dir + "/%s_%s_s%d_trace.out.gz"%\
                                  (db["gp_full_name"][item], sub_item, i)
                            training_files.append(result_file)
                    else:
                        if (sp_stat['sub_item'] == 'Unknown'):
                            sp_stat['sub_item'] = sub_item
                            sp_stat['num'] = 1
                        if (sp_stat['sub_item'] == sub_item):
                            if (sp_stat['num'] <= num_spts):
                                result_dir = trace_dir + \
                                     "/%s/%s/simpoint_%d" % (item, sub_item, sp_stat['num'])
                                result_file = result_dir + "/%s_%s_s%d_trace.out.gz"%\
                                      (db["gp_full_name"][item], sub_item, sp_stat['num'])
                                training_files.append(result_file)
                                break
                            else:
                                sp_stat['sub_item'] = 'Unknown'
                                continue
        if (sp_stat['sub_item'] == 'Unknown'):
            sp_stat['finished'] = True
        j.close()
        del db
        return training_files, spt_count

    training_files, spt_count = get_train_file_name()
    train_data = []
    eval_data = []
    for file_name in training_files:
        cur_work_group = file_name.split('/')[4]
        logging.info("Loading data: " + file_name)
        with gzip.open(file_name, 'rt') as f:
            lines = f.readlines()
            trace_per_spt = config.workgroup_trace_length // spt_count[cur_work_group]
            if (len(lines) > trace_per_spt):
                lines = lines[:trace_per_spt]
            else:
                lines += lines[:trace_per_spt - len(lines)]
            for i, line in enumerate(lines):
                pline = i, int(line, 16)
                if i < train_split * len(lines):
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    return train_data, eval_data

#%%% My Functions

def split_to_words(value,BN_bits,split_bits):
    res=[SPLITER_ID]
    for i in range(BN_bits//split_bits+1):
        divider=2**split_bits
        res.append(value%(divider)+OFFSET)#add 1, range(1-64),0 as padding
        value=value//divider
    return res


def create_window(df):
    for i in range(PRED_FORWARD):
        df['words_future_%d'%(i+1)]=df['words'].shift(periods=-(i+1))
    for i in range(LOOK_BACK):
        df['words_past_%d'%(i+1)]=df['words'].shift(periods=(i+1))
    return df


def concact_past_future(df):
    for i in range(PRED_FORWARD):
        if i==0:
            df["future"]=df['words_future_%d'%(i+1)]
        else:
            df["future"]+=df['words_future_%d'%(i+1)]
    for i in range(LOOK_BACK):
        if i==0:
            df["past"]=df['words_past_%d'%(i+1)]
        else:
            df["past"]+=df['words_past_%d'%(i+1)]
    return df


def add_start_end(column_list, start_id, end_id):
    return [start_id] + column_list + [end_id]

#%% Main of Transformer

#%%% NoamOpt

class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


#%%% Run
def run(df_train,df_test,model_save_path,log_path,load=False):

    train_dataset = MAPDataset(df_train)
    test_dataset = MAPDataset(df_test)

    logging.info("-------- Dataset Build! --------")
    train_dataloader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=
                                True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=
                              True,collate_fn=train_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # Initialize model
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_encoder_layers, config.n_decoder_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    # model_par = torch.nn.DataParallel(model)
    if load==True:
        model.load_state_dict(torch.load(model_save_path))
    model_par=model
    # Train
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx,
                                   smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    logging.info(summary(model, verbose=0))
    logging.info("Training Start:")
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer,model_save_path)
    logging.info("-------- Training Completed! --------")


def one_access_predict(sent,model_save_path, beam_search=False):
    # Initialize model
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_encoder_layers, config.n_decoder_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    batch_input = torch.LongTensor(np.array(sent)).to(config.device)
    return translate(batch_input, model, model_save_path, use_beam=beam_search)


def predict_case(df_case,idx,model_save_path):
    sent = list(df_case["past"][idx:idx+1].values)
    return one_access_predict(sent,model_save_path,beam_search=False)


def print_single_predict(df_case,idx,model_save_path):
    print("input:")
    print(df_case["past"][idx:idx+1].values[0])
    label=df_case["future"][idx:idx+1].values
    print("label:")
    print(label[0][1:-1])

    pred=predict_case(df_case,idx,model_save_path)
    print("predict:")
    print(pred[0])
    return pred


def convert_to_binary(data,bit_size=64-6):
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    res=get_bin(data,bit_size)
    return [int(char)+OFFSET for char in res]
    # make it (1,2)


#bitmap(1,2)
def to_bitmap(n,bitmap_size):
    l0=np.ones((bitmap_size),dtype = int)
    if(len(n)>0):
        for x in n:
            l0[int(x)]=1+OFFSET
        l1=list(l0)
        #l1.reverse()
        return l1
    else:
        return list(l0)
    #print("Bitmap completed")


def preprocessing_bit(train_data):
    df = pd.DataFrame(train_data)
    df.columns = ["id", "addr"]
    df['raw'] = df['addr']
    df['page_address'] = [x >> PAGE_BITS for x in df['raw']]
    df['page_offset'] = [x - (x >> PAGE_BITS << PAGE_BITS) for x in df['raw']]
    df['cache_line_index'] = [int(x >> BLOCK_BITS) for x in df['page_offset']]
    df['page_cache_index'] = [x >> BLOCK_BITS for x in df['raw']]

    df["page_cache_index_bin"] = df.apply(
            lambda x: convert_to_binary(x['page_cache_index'], BLOCK_NUM_BITS), axis=1)

    # past
    for i in range(LOOK_BACK):
        df['page_cache_index_bin_past_%d' % (i+1)] = df['page_cache_index_bin'].shift(periods=(i+1))
        if i == 0:
            df["past"] = df['page_cache_index_bin_past_%d' % (i+1)]
        else:
            df["past"] += df['page_cache_index_bin_past_%d' % (i+1)]

    # labels
    df = df.sort_values(by=["page_address", "id"])
    for i in range(PRED_FORWARD):
        shift_period = -(i + 1) - SKIP_FORWARD
        df['cache_line_index_future_%d' % (i+1)] = df['cache_line_index'].shift(periods=shift_period)

    for i in range(PRED_FORWARD):
        if i == 0:
            df["future_idx"] = df['cache_line_index_future_%d' % (i+1)]
        else:
            df["future_idx"] = df[['future_idx', 'cache_line_index_future_%d' % (i+1)]].values.astype(int).tolist()

    df = df.sort_values(by=["id"])
    df = df.dropna()

    df["future"] = (np.stack(df["future_idx"])+OFFSET).tolist()

    # df["future"]=df.apply(lambda x: to_bitmap(x['future_idx'],BITMAP_SIZE),axis=1)
    df["future"] = df.apply(lambda x: add_start_end(x['future'], START_ID, END_ID), axis=1)

    df["past"] = df.apply(lambda x: add_start_end(x['past'], START_ID, END_ID), axis=1)

    return df


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore')
    import sys

    json_path = sys.argv[1]
    trace_dir = sys.argv[2]
    model_save_path = sys.argv[3]
    WORK_GROUP = sys.argv[4]
    TRAIN_SPLIT = float(sys.argv[5])
    GPU_NUM = sys.argv[6]
    if (sys.argv[7] == 'sp_mode'):
        SIMPOINT_MODE = True
    else:
        SIMPOINT_MODE = False

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

    if os.path.isfile(model_save_path):
        loading = True
        if (SIMPOINT_MODE is True):
            print("[Error] Simpoint mode specified but model file found.")
            exit(-1)
    else:
        loading = False
        if (SIMPOINT_MODE is True):
            os.mkdir(model_save_path)
            log_path = "%s/%s.log" % (model_save_path, model_save_path)
        else:
            log_path = model_save_path + ".log"

    print("TransforMAP training start, loading:", loading)
    utils.set_logger(log_path)
    logging.info("history length: " + str(config.LOOK_BACK))
    logging.info("prefetch degree: " + str(config.PRED_FORWARD))
    logging.info("work group: " + WORK_GROUP)
    logging.info("trace path: " + trace_dir)
    logging.info("=======================================")
    logging.info("d_model: " + str(config.d_model))
    logging.info("d_ff: " + str(config.d_ff))
    logging.info("n_heads: " + str(config.n_heads))
    logging.info("n_encoder_layers: " + str(config.n_encoder_layers))
    logging.info("n_decoder_layers: " + str(config.n_decoder_layers))
    if (SIMPOINT_MODE is True):
        sp_stat = {
            'num': 0,
            'sub_item': "Unknown",
            'finished': False
        }
        while (True):
            train_data, eval_data = read_load_trace_data(json_path,
                                                         trace_dir,
                                                         WORK_GROUP,
                                                         TRAIN_SPLIT,
                                                         sp_stat=sp_stat)
            if (sp_stat['finished'] is True):
                break
            df_train = preprocessing_bit(train_data)[:][["future", "past"]]
            Len_test = len(eval_data) if len(eval_data) < 10000 else 10000
            df_test = preprocessing_bit(eval_data)
            df_test = df_test[:Len_test][["future", "past"]]
            sp_save_path = model_save_path + '/%s_sp%d.pt' % (sp_stat['sub_item'], sp_stat['num'])
            run(df_train, df_test, sp_save_path, log_path, load=loading)
            print_single_predict(df_test, 1090, sp_save_path)
            sp_stat['num'] += 1
    else:
        train_data, eval_data = read_load_trace_data(json_path, trace_dir,
                                                     WORK_GROUP, TRAIN_SPLIT)
        df_train = preprocessing_bit(train_data)[:][["future", "past"]]
        Len_test = len(eval_data) if len(eval_data) < 10000 else 10000
        df_test = preprocessing_bit(eval_data)[:Len_test][["future", "past"]]
        run(df_train, df_test, model_save_path, log_path, load=loading)
        print_single_predict(df_test, 1090, model_save_path)
