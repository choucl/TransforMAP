import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
config.device = torch.device('cpu')

from TMAP_train import read_load_trace_data, preprocessing_bit
from TMAP_train import make_model
from model import batch_greedy_decode
from data_loader import MAPDataset

background_colors = ['green', 'yellow']

def save_attention_figs(transformer_layer, save_dir_path):

    def iterate_heads(property, name):
        for i, head_result in tqdm(enumerate(property)):
            plt.rcParams["figure.figsize"] = (10, 10)
            fig, ax = plt.subplots()
            for x in range(config.LOOK_BACK):
                plt.axvspan(1 + x * 58, 1 + (x + 1) * 58 - 6,
                            color=background_colors[x % 2], alpha=0.1)
                plt.axvspan(1 + (x + 1) * 58 - 6, 1 + (x + 1) * 58,
                            color=background_colors[x % 2], alpha=0.3)
            _ = ax.imshow(head_result, cmap="Blues")
            plt.savefig(save_dir_path + '/%s_%d.png' % (name, i))

    iterate_heads(transformer_layer.self_attn.attn[0], 'attn')
    iterate_heads(transformer_layer.self_attn.attn_origin[0], 'attn_origin')
    iterate_heads(transformer_layer.self_attn.attn_scores[0], 'attn_score')


def visualizing(df_data, model_save_path, img_save_path):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_encoder_layers,
                       config.n_decoder_layers, config.d_model, config.d_ff, config.n_heads,
                       config.dropout)
    model.load_state_dict(torch.load(model_save_path))
    model = model.eval()

    res = []
    dataset = MAPDataset(df_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=dataset.collate_fn)
    with torch.no_grad():
        for batch in dataloader:
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            decode_result = batch_greedy_decode(model, src, src_mask,
                                                max_len=config.max_len)
            save_attention_figs(model.encoder.layers[0], img_save_path)
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
    img_save_path = sys.argv[5]

    if os.path.isfile(model_save_path) :
       loading=True
    else:
       loading=False

    print("TransforMAP visualize start, loading:", loading)
    read_data, _ = read_load_trace_data(json_path, trace_dir, WORK_GROUP, 1)
    df_data = preprocessing_bit(read_data)[:1][["past", "future"]]
    visualizing(df_data, model_save_path, img_save_path)
