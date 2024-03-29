import torch
import random
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
front_colors = ['Blues', 'Reds', 'Greys', 'Greens']

attn = None
attn_origin = None

def update_attention_figs(transformer_layer):
    global attn
    global attn_origin
    if (attn is None):
        attn = transformer_layer.self_attn.attn[0].clone().detach()
    else:
        attn += transformer_layer.self_attn.attn[0]

    if (attn_origin is None):
        attn_origin = transformer_layer.self_attn.attn_origin[0].clone().detach()
    else:
        attn_origin += transformer_layer.self_attn.attn_origin[0]


def save_attention_figs(save_dir_path, visualize_mode, model_name):

    def iterate_heads(property, name):
        if (visualize_mode == 'fuse'):
            print("Saving figures: %s in fuse mode" % name)
            plt.figure()
            plt.rcParams["figure.figsize"] = (10, 10)
            for x in range(config.LOOK_BACK):
                plt.axvspan(1 + x * 58, 1 + (x + 1) * 58 - 6,
                            color=background_colors[x % 2], alpha=0.05)
                plt.axvspan(1 + (x + 1) * 58 - 6, 1 + (x + 1) * 58,
                            color=background_colors[x % 2], alpha=0.15)
            for i, head_result in enumerate(property):
                head_result = head_result.numpy()
                plt.imshow(head_result, cmap=front_colors[i],
                           alpha=head_result / head_result.max(),
                           interpolation='none')
            plt.savefig(save_dir_path + '/%s.png' % (name))
        else:
            print("Saving figures: %s in seperate mode" % name)
            for i, head_result in enumerate(property):
                plt.figure()
                plt.rcParams["figure.figsize"] = (10, 10)
                for x in range(config.LOOK_BACK):
                    plt.axvspan(1 + x * 58, 1 + (x + 1) * 58 - 6,
                                color=background_colors[x % 2], alpha=0.05)
                    plt.axvspan(1 + (x + 1) * 58 - 6, 1 + (x + 1) * 58,
                                color=background_colors[x % 2], alpha=0.15)
                plt.imshow(head_result, cmap=front_colors[i],
                           interpolation='none')
                plt.savefig(save_dir_path + '/%s_%d.png' % (name, i))

    iterate_heads(attn, model_name + '_attn')
    iterate_heads(attn_origin, model_name + '_attn_origin')
    print("Done saving figs")


def visualizing(df_data, model_save_path, img_save_path, visualize_mode):
    model = make_model(config.src_vocab_size, config.tgt_vocab_size,
                       config.n_encoder_layers, config.n_decoder_layers,
                       config.d_model, config.d_ff, config.n_heads,
                       config.dropout)
    model.load_state_dict(torch.load(model_save_path))
    model = model.eval()

    res = []
    dataset = MAPDataset(df_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=dataset.collate_fn)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            decode_result = batch_greedy_decode(model, src, src_mask,
                                                max_len=config.max_len)
            update_attention_figs(model.encoder.layers[0])
            res.extend(decode_result)
    res = np.array(res)[:, :-1]
    model_name = model_save_path.split('/')[-1]
    save_attention_figs(img_save_path, visualize_mode, model_name)


if __name__ == "__main__":

    import os
    import warnings
    warnings.filterwarnings('ignore')
    import sys

    json_path = sys.argv[1]
    trace_dir = sys.argv[2]
    model_save_path = sys.argv[3]
    WORK_GROUP = sys.argv[4]
    img_save_path = sys.argv[5]
    if (len(sys.argv) > 6):
        visualize_mode = sys.argv[6]
    else:
        visualize_mode = "fuse"

    if os.path.isfile(model_save_path):
        loading = True
    else:
        loading = False

    print("TransforMAP visualize start, loading:", loading)
    read_data, _ = read_load_trace_data(json_path, trace_dir, WORK_GROUP, 1)
    df_data = preprocessing_bit(read_data).sample(n=1000)[["past", "future"]]
    visualizing(df_data, model_save_path, img_save_path, visualize_mode)
