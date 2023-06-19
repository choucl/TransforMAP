'''
This file will convert the original pt file to the script module pt
Usage: python3 convert.py original_pt_path
'''

import torch
import sys
import config
from model import make_model

DEVICE = torch.device("cuda")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    original_pt_paths = sys.argv[1:]
    if (original_pt_paths == ""):
        print("This file will convert the original pt file to the script module pt")
        print("Usage: python3 convert.py original_pt_path")
        exit(-1)
    for path in original_pt_paths:
        model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                           config.d_model, config.d_ff, config.n_heads, config.dropout)
        model.to(DEVICE)
        model.load_state_dict(torch.load(path))
        src_size = config.BLOCK_NUM_BITS * config.LOOK_BACK + 2
        example_src = torch.rand(config.batch_size, src_size).to(torch.int).to(DEVICE)
        example_tgt = torch.rand(config.batch_size, 3).to(torch.int).to(DEVICE)
        example_src_mask = torch.rand(config.batch_size, 1, src_size).to(torch.int).to(DEVICE)
        example_src_tgt = torch.rand(config.batch_size, 3, 3).to(torch.int).to(DEVICE)
        model.eval().to(DEVICE)
        traced_script_module = torch.jit.trace(model, (example_src, example_tgt, example_src_mask, example_src_tgt))
        traced_script_module.save(path.split(".")[0] + "_smodule.pt")
        gen_src = torch.rand(1, config.d_model).to(DEVICE)
        gen_script_module = torch.jit.trace(model.generator, gen_src)
        gen_script_module.save(path.split(".")[0] + "_gen_smodule.pt")

if __name__ == "__main__":
    main()
