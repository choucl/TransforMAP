import json
import gzip
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

import config
from TMAP_train import read_load_trace_data, preprocessing_bit
from TMAP_train import make_model
from train import translate
from model import batch_greedy_decode
from data_loader import MAPDataset

CACHE_LINE_WAY = 4

def process(idx, pred, df_data):
    hit_count = 0
    issued_prefetch = len(np.unique(pred))
    pred = pred - config.OFFSET
    cur_page = df_data["page_address"][idx:idx+1].values[0]
    for future_data_idx in range(idx + 1, len(df_data.index)):
        future_page = df_data["page_address"][future_data_idx:future_data_idx+1].values[0]
        future_block = df_data["cache_line_index"][future_data_idx:future_data_idx+1].values[0]
        way_count = 0
        if (future_block in pred):
            way_count += 1
            if (way_count == CACHE_LINE_WAY):
                print(idx, ": Evicted")
                break
            elif (future_page == cur_page):
                print(idx, ": hit", future_page, future_block)
                way_count = 0
                hit_count += 1
    return issued_prefetch, hit_count

def testing(df_data, model_save_path):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.load_state_dict(torch.load(model_save_path))
    model = model.eval()

    res = []
    dataset = MAPDataset(df_data)
    dataloader = DataLoader(dataset, batch_size=1024,
                            shuffle=False, collate_fn=dataset.collate_fn)
    print(model.training)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            decode_result = batch_greedy_decode(model, src, src_mask,
                                                max_len=config.max_len)
            res.extend(decode_result)
    res = np.array(res)[:, :-1]
    print(res[0])

    test_results = Parallel(n_jobs=30)(delayed(process)(i, res[i], df_data) for i in range(res.shape[0]))
    test_results = np.array(test_results).sum(axis=0)
    print(test_results)

    print("Issued prefetch:", test_results[0])
    print("Hit count:", test_results[1])

if __name__ == "__main__":

    import os
    import warnings
    warnings.filterwarnings('ignore')
    import sys

    json_path=sys.argv[1]
    trace_dir=sys.argv[2]
    model_save_path=sys.argv[3]
    WORK_GROUP = sys.argv[4]
    GPU_NUM = sys.argv[5]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

    if os.path.isfile(model_save_path) :
       loading=True
    else:
       loading=False

    print("TransforMAP testing start, loading:", loading)
    read_data, _ = read_load_trace_data(json_path, trace_dir, WORK_GROUP, 1)
    df_data = preprocessing_bit(read_data)[:][["past", "future", "id", "page_address", "cache_line_index"]]
    testing(df_data, model_save_path)
