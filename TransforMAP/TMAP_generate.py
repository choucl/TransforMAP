#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 02:36:18 2021

@author: pengmiao
"""
import sys
import utils
import config
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MAPDataset
from model import make_model, LabelSmoothing
import pdb
from tqdm import tqdm
import lzma

#%%%
BLOCK_BITS=config.BLOCK_BITS
TOTAL_BITS=config.TOTAL_BITS
SPLIT_BITS=config.SPLIT_BITS
VOCAB_SIZE=2**SPLIT_BITS
LOOK_BACK=config.LOOK_BACK
PRED_FORWARD=config.PRED_FORWARD

PAD_ID=config.PAD_ID
START_ID=config.START_ID
END_ID=config.END_ID
BLOCK_NUM_BITS=config.BLOCK_NUM_BITS
PAGE_BITS=config.PAGE_BITS
BITMAP_SIZE=config.BITMAP_SIZE
OFFSET=config.OFFSET
#%%% Interface

def read_load_trace_data(load_trace, num_prefetch_warmup_instructions):
    
    def process_line(line):
        split = line.strip().split(', ')
        return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

    train_data = []
    eval_data = []
    if file_path[-2:] == 'xz':
        with lzma.open(load_trace, 'rt') as f:
            for line in f:
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)
    else:
        with open(load_trace, 'r') as f:
            for line in f:
                pline = process_line(line)
                if pline[0] < num_prefetch_warmup_instructions * 1000000:
                    train_data.append(pline)
                else:
                    eval_data.append(pline)

    return train_data, eval_data

#%% Preprocessing

def split_to_words(value,BN_bits,split_bits):
    res=[SPLITER_ID]
    for i in range(BN_bits//split_bits+1):
        divider=2**split_bits
        res.append(value%(divider)+4)#add 1, range(1-64),0 as padding
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


def add_start_end(column_list,start_id,end_id):
    return [start_id]+column_list+[end_id]

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

def preprocessing_bit(data,TOTAL_NUM):
    df=pd.DataFrame(data)
    df.columns=["id", "cycle", "addr", "ip", "hit"]
    if TOTAL_NUM != None:
        df=df[df["id"]<TOTAL_NUM*1000000]
    df['raw']=df['addr']
    df['page_address'] = [ x >> PAGE_BITS for x in df['raw']]
    #df['page_address_str'] = [ "%d" % x for x in df['page_address']]
    df['page_offset'] = [x- (x >> PAGE_BITS<<PAGE_BITS) for x in df['raw']]
    df['cache_line_index'] = [int(x>> BLOCK_BITS) for x in df['page_offset']]
    df['page_cache_index'] = [x>>BLOCK_BITS for x in df['raw']]
    
    df["page_cache_index_bin"]=df.apply(lambda x: convert_to_binary(x['page_cache_index'],BLOCK_NUM_BITS),axis=1)
    
    # past
    for i in range(LOOK_BACK):
        df['page_cache_index_bin_past_%d'%(i+1)]=df['page_cache_index_bin'].shift(periods=(i+1))
        
    for i in range(LOOK_BACK):
        if i==0:
            df["past"]=df['page_cache_index_bin_past_%d'%(i+1)]
        else:   
            df["past"]+=df['page_cache_index_bin_past_%d'%(i+1)]
    
    # labels
    df=df.sort_values(by=["page_address","cycle"])
    for i in range(PRED_FORWARD):
        df['cache_line_index_future_%d'%(i+1)]=df['cache_line_index'].shift(periods=-(i+1))
    
    for i in range(PRED_FORWARD):
            if i==0:
                df["future_idx"]=df['cache_line_index_future_%d'%(i+1)]
            else:   
                df["future_idx"] = df[['future_idx','cache_line_index_future_%d'%(i+1)]].values.astype(int).tolist()
    
    df=df.sort_values(by=["id"])
    df=df.dropna()
    
    df["future"]=(np.stack(df["future_idx"])+OFFSET).tolist()
    
 #   df["future"]=df.apply(lambda x: to_bitmap(x['future_idx'],BITMAP_SIZE),axis=1)
    df["future"]=df.apply(lambda x: add_start_end(x['future'],START_ID,END_ID),axis=1)

    df["past"]=df.apply(lambda x: add_start_end(x['past'],START_ID,END_ID),axis=1)

    return df

#%% All_prediction batch by batch
#
def Prediction(dataframe,model_save_path):
    prediction=[]

    test_dataset = MAPDataset(dataframe)

    print("-------- Dataset Build! --------")
    test_data=DataLoader(test_dataset,batch_size=config.batch_size,shuffle= False,collate_fn=test_dataset.collate_fn)

    print("-------- Get Dataloader! --------")

    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
 
    for batch in tqdm(test_data):
        res=translate(batch.src, model, model_save_path,use_beam=False)#(batch,[length])
        prediction.extend(res)
    dataframe["predicted"]= prediction#(all,[length])
    
    return dataframe

def split_to_words(value,BN_bits,split_bits):
    res=[SPLITER_ID]
    for i in range(BN_bits//split_bits+1):
        divider=2**split_bits
        res.append(value%(divider)+4)#add 1, range(1-64),0 as padding
        value=value//divider
    return res
#%% Convert back to absolute address
def words_back_address(values,BN_bits,split_bits):
    multiplier=2**SPLIT_BITS
    res=[]
    period=BN_bits//split_bits+1+1 #11
    for i in range(PRED_FORWARD):
        res_1=0
        for j in range(period-1):
            res_1 +=(multiplier**j)*(values[i*period+j+1]-4)
        res.append(res_1)
    return res


def convert_to_raw_hex(pred_addr,block_bits):
    res=int(int(pred_addr)<<block_bits)
    pred_raw_hex=res.to_bytes(((res.bit_length()+7)//8),"big").hex().lstrip('0')
    return pred_raw_hex


def preprocessing(data,TOTAL_NUM):  
    df=pd.DataFrame(data)
    #df_eval=pd.DataFrame(eval_data)
    df.columns=["id", "cycle", "addr", "ip", "hit"]
    if TOTAL_NUM != None:
        df=df[df["id"]<TOTAL_NUM*1000000]
    #df['raw'] = [int (x,16) for x in df['addr']]
    df['raw']=df["addr"]
    df['b_raw'] = [x>> BLOCK_BITS for x in df['raw']]
    df['b_raw_x'] = ["%d" % (x) for x in df['b_raw']]
    
    df['words']=df.apply(lambda x: split_to_words(x['b_raw'],BLOCK_NUM_BITS,SPLIT_BITS),axis=1)
    
    df=create_window(df)
    df=df.dropna()
    df=concact_past_future(df)
    df["future"]=df.apply(lambda x: add_start_end(x['future'],START_ID,END_ID),axis=1)
    df["past"]=df.apply(lambda x: add_start_end(x['past'],START_ID,END_ID),axis=1)
    return df

def post_processing(df,buffer_size):   
    df['pred_addr']=df.apply(lambda x: words_back_address(x['predicted'],BLOCK_NUM_BITS,SPLIT_BITS),axis=1)
    df_case=df[["id","b_raw","pred_addr"]]
    Buffer=[]
    real_pred=[]
    real_pred_all=[]
    is_predicted=[]
    for index,rows in tqdm(df_case.iterrows()):
        Buffer.append(rows["b_raw"])
        useful=list(set(rows["pred_addr"])-set(Buffer))
        real_pred=useful[:2]
        if len(real_pred)==1:
            real_pred.append(rows["b_raw"]+1)
        elif len(real_pred)==0:
            real_pred.append(rows["b_raw"]+1)
            real_pred.append(rows["b_raw"]+2)
        real_pred_all.append(real_pred)
        Buffer.extend(real_pred)
        Buffer=Buffer[-BUFFER_SIZE:]
        is_predicted.append(not set(rows["pred_addr"]).isdisjoint(real_pred))
    
    df_case["real_pred"]=real_pred_all
    df_case["is_predicted"]=is_predicted
    
    df_case=df_case[["id","real_pred"]].explode("real_pred")
    df_case['pred_addr_raw_hex']=df_case.apply(lambda x: convert_to_raw_hex(x["real_pred"], BLOCK_BITS),axis=1)
    return df_case[["id","pred_addr_raw_hex"]]

def add_n_convert(pred_2,page_address):
    res=int(((int(page_address)<<BLOCK_BITS) + int(pred_2))<<BLOCK_BITS)
    res2=res.to_bytes(((res.bit_length() + 7) // 8),"big").hex().lstrip('0')
    return res2
   
def post_processing_bit(df,offset):
    df["pred_index"]=(np.stack(df["predicted"])-offset).tolist()
    df=df.explode('pred_index')
    df=df.dropna()
    df['pred_hex'] = df.apply(lambda x: add_n_convert(x['pred_index'], x['page_address']), axis=1)
    df_res=df[["id","pred_hex"]]
    return df_res
        
    
#%% Main
import pdb
#pdb.set_trace()

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'2 in tarim'
    import warnings
    warnings.filterwarnings('ignore')

    import sys
    file_path=sys.argv[1]
    model_save_path=sys.argv[2]
    path_to_prefetch_file=sys.argv[3]

    if len(sys.argv)==6:
        TRAIN_NUM = int(sys.argv[4])
        TOTAL_NUM = int(sys.argv[5])
    elif len(sys.argv)==5:
        TRAIN_NUM = int(sys.argv[4])
        TOTAL_NUM=None
    else:
        TRAIN_NUM=0
        TOTAL_NUM=None

    print("TranforMAP generating start")
    train_data, eval_data = read_load_trace_data(file_path, TRAIN_NUM)      
    df=preprocessing_bit(eval_data,TOTAL_NUM)
    df=Prediction(df,model_save_path)
    df_res=post_processing_bit(df,OFFSET)
    df_res.to_csv(path_to_prefetch_file,header=False, index=False, sep=" ")



































