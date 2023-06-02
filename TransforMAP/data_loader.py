import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import config
DEVICE = config.device
PAD_ID=config.PAD_ID

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, trg=None,pad=0):
        src = src.to(DEVICE)
        self.src=src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    # Mask
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    
class MAPDataset(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
        self.future=list(df["future"].values)

    def __getitem__(self, idx):
        
        past = self.past[idx]
        future = self.future[idx]
        return [past, future]

    def __len__(self):
        return len(self.past)

    def collate_fn(self, batch):
        
        past_b = [x[0] for x in batch]
        future_b = [x[1] for x in batch]
        
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in past_b],
                                   batch_first=True, padding_value=PAD_ID)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in future_b],
                                    batch_first=True, padding_value=PAD_ID)
        return Batch(batch_input,batch_target,PAD_ID)