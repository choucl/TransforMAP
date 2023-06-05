'''
This file will convert the original pt file to the script module pt
Usage: python3 convert.py original_pt_path
'''

import torch
import sys
import config
from model import make_model

def main():
    original_pt_path = sys.argv[1]
    if (original_pt_path == ""):
        print("This file will convert the original pt file to the script module pt")
        print("Usage: python3 convert.py original_pt_path")
        exit(-1)
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.to(config.device)
    model.load_state_dict(torch.load(original_pt_path))
    src_size = config.BLOCK_NUM_BITS * config.LOOK_BACK + config.PRED_FORWARD
    example_src = torch.rand(config.batch_size, src_size).to(torch.int).to(config.device)
    example_tgt = torch.rand(config.batch_size, 3).to(torch.int).to(config.device)
    example_src_mask = torch.rand(config.batch_size, 1, src_size).to(torch.int).to(config.device)
    example_src_tgt = torch.rand(config.batch_size, 3, 3).to(torch.int).to(config.device)
    traced_script_module = torch.jit.trace(model, (example_src, example_tgt, example_src_mask, example_src_tgt))
    traced_script_module.save(original_pt_path.split(".")[0] + "_smodule.pt")

if __name__ == "__main__":
    main()
