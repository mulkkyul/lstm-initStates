import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import glob, os

class myDataset(Dataset):
    def __init__(self, data):
        self.data_input, self.data_label = data
    def __getitem__(self, index):
        return self.data_input[index], self.data_label[index]
    def __len__(self):
        return len(self.data_input)

DATA_TR = []
DATA_INPUT = []
DATA_LABEL = []

# ================================================================================================
# These variables are used in other python files as well
batchSize = 1024
numSequences = 0
seqLenth = 241 # Length of the sequence.
# ================================================================================================

print('=' * 80)
print('Loading Dataset....')

txtFiles = glob.glob("./data/2019-01-01_2019-12-31/*KS*.txt")
txtFiles = sorted(txtFiles)
numFiles = len(txtFiles)

maxLength  = 0

for idx in range(numFiles):
    posFilename = txtFiles[idx]
    data = torch.Tensor(np.loadtxt(posFilename))

    scalingFactor = data[0]
    data = data / scalingFactor
    data = data - 1.0 

    DATA_INPUT.append(data)
    DATA_LABEL.append([idx])
    numSequences += 1

DATA_TR.append(DATA_INPUT)
DATA_TR.append(DATA_LABEL)


def pad_collate(batch):
  (xx, yy) = zip(*batch)
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

  return xx_pad, yy


dataset = myDataset(DATA_TR)
data_loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, collate_fn=pad_collate)
numBatches = len(data_loader)
print('...................................................................... Completed')

print('Number of data: %d' % numSequences)
print('Batch Size: %d' % batchSize)
print('Number of Batches: %d' % numBatches)
print('-' * 80)



