import torch

BLOCK_BITS=6
PAGE_BITS=12
TOTAL_BITS=64
BLOCK_NUM_BITS=TOTAL_BITS-BLOCK_BITS
SPLIT_BITS=6
LOOK_BACK=5
PRED_FORWARD=2
BITMAP_SIZE=2**(PAGE_BITS-BLOCK_BITS)


PAD_ID=0
START_ID=2
END_ID=3
VOCAB_SIZE=64+3 #binary+3
OFFSET=4

d_model = 128#512
n_heads = 4#8
n_layers = 4#6
d_k = 32#64
d_v = 32#64
d_ff = 128#2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 64+4#32000 #spliter,pad,end,start
tgt_vocab_size = 64+4#32000
batch_size = 128
epoch_num = 50
early_stop = 3
lr = 3e-4


#max_len = PRED_FORWARD*BITMAP_SIZE+2-1#60
max_len = PRED_FORWARD*1+2-1#60
# beam size for bleu
beam_size = 2
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

gpu_id = '0'
#device_id = [0, 1]
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
    
