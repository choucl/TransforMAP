'''
This file will convert the original pt file to the script module pt
Usage: python3 convert.py original_pt_path
'''

import torch
import sys
import config
from model import make_model

CPU_DEVICE = torch.device("cpu")

def main():
    original_pt_path = sys.argv[1]
    if (original_pt_path == ""):
        print("This file will convert the original pt file to the script module pt")
        print("Usage: python3 convert.py original_pt_path")
        exit(-1)
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.to(CPU_DEVICE)
    model.load_state_dict(torch.load(original_pt_path))
    src_size = config.BLOCK_NUM_BITS * config.LOOK_BACK + 2
    example_src = torch.rand(config.batch_size, src_size).to(torch.int).to(CPU_DEVICE)
    example_tgt = torch.rand(config.batch_size, 3).to(torch.int).to(CPU_DEVICE)
    example_src_mask = torch.rand(config.batch_size, 1, src_size).to(torch.int).to(CPU_DEVICE)
    example_src_tgt = torch.rand(config.batch_size, 3, 3).to(torch.int).to(CPU_DEVICE)
    model.eval()
    traced_script_module = torch.jit.trace(model, (example_src, example_tgt, example_src_mask, example_src_tgt))
    traced_script_module.save(original_pt_path.split(".")[0] + "_smodule.pt")

    gen_src = torch.rand(1, config.d_model).to(CPU_DEVICE)
    gen_script_module = torch.jit.trace(model.generator, gen_src)
    gen_script_module.save(original_pt_path.split(".")[0] + "_gen_smodule.pt")

if __name__ == "__main__":
    main()
