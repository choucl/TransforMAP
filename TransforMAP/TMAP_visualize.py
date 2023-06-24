import json
import gzip
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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

def visualizing(df_data, model_save_path):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_encoder_layers, config.n_decoder_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.load_state_dict(torch.load(model_save_path))
    model = model.eval()

    res = []
    dataset = MAPDataset(df_data)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, collate_fn=dataset.collate_fn)
    print(model.training)
    with torch.no_grad():
        for batch in dataloader:
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            decode_result = batch_greedy_decode(model, src, src_mask,
                                                max_len=config.max_len)
            for i, head in tqdm(enumerate(model.encoder.layers[0].self_attn.attn[0])):
                plt.rcParams["figure.figsize"] = (10, 10)
                scores = head
                fig, ax = plt.subplots()
                im = ax.imshow(scores)
                plt.savefig('png/' + str(i) + '_attn.png')
            res.extend(decode_result)
    res = np.array(res)[:, :-1]


if __name__ == "__main__":

    import os
    import warnings
    warnings.filterwarnings('ignore')
    import sys

    json_path=sys.argv[1]
    trace_dir=sys.argv[2]
    model_save_path=sys.argv[3]
    WORK_GROUP = sys.argv[4]

    if os.path.isfile(model_save_path) :
       loading=True
    else:
       loading=False

    print("TransforMAP visualize start, loading:", loading)
    read_data, _ = read_load_trace_data(json_path, trace_dir, WORK_GROUP, 1)
    df_data = preprocessing_bit(read_data)[:1][["past", "future"]]
    visualizing(df_data, model_save_path)
