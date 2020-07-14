import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import myDataloader as db

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.numLayers = 1
        self.input_dim = 1
        self.output_dim = self.input_dim
        self.hidden_size = 128
        self.batchSize = db.batchSize  # or numData when numData < batchSize
        self.numData = db.numSequences
        if(self.numData <= self.batchSize):
            self.batchSize = self.numData

        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_dim)

        h0 = torch.zeros(self.numLayers, self.numData, self.hidden_size).to(device)
        self.h0 = nn.Parameter(h0, requires_grad=True)


    def forward(self, input, label):
        seqLength = input.shape[0]
        numDataInBatch = input.shape[1]
        for i in range(len(label)):
            idxSeq = label[i][0]
            temp = self.h0[:,idxSeq,:].view(1,1,-1)
            if(i == 0):
                h0 = temp
            else:
                h0 = torch.cat((h0, temp), 1)

        for t in range(seqLength):
            if(t ==0):
                h_t = h0
                c_t = torch.zeros(self.numLayers, numDataInBatch, self.hidden_size).to(device)
                input_t = torch.zeros(1, numDataInBatch, self.input_dim).to(device)  # firstDim = time = 1
            else:
                input_t = output

            output, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))

            output = self.linear(h_t)

            if (t == 0):
                outputs = output
            else:
                outputs = torch.cat((outputs,output),0)

        return outputs


if __name__ == "__main__":

    trainModel = False # True for train, False for evaluation
    if(len(sys.argv) != 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'eval')):
        print('Please specify the mode.\ne.g. python lstm-initStates.py train  or python lstm-initStates.py check')
        exit()

    if(sys.argv[1] == 'train'):
        learningRate = 0.001
        numEpoch = 100000

        model = Sequence().to(device)
        optimizer = Adam(params=model.parameters(), lr=learningRate)
        criterion = nn.MSELoss()


        fid = open('./result/loss.txt','w')
        minLoss = 0.001

        for epoch in range(numEpoch):
            for index, (nn_in, nn_label) in enumerate(db.data_loader):
                nn_in = nn_in.to(device)
                train_data = nn_in.permute(1, 0)  # (seqLength, batchsize)
                train_data = train_data.view(db.seqLenth, -1, model.output_dim)
                output = model.forward(train_data, nn_label)
                optimizer.zero_grad()
                loss = criterion(output, train_data)
                loss = loss.to(device)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                loss_val = loss.detach().cpu().numpy()
                print(f'Epoch {epoch}, loss: {loss_val:.8f}')
                fid.write('%d %.8f\n'%(epoch, loss_val))
                fid.flush()
                if (minLoss > loss_val):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, './result/model_best.tar')

        fid.close()
    elif (sys.argv[1] == 'eval'):
        if(torch.cuda.is_available()):
            checkpoint = torch.load('./result/model_best.tar')
        else:
            checkpoint = torch.load('./result/model_best.tar', map_location=torch.device('cpu'))

        learningRate = 0.001
        numEpoch = 100000

        model = Sequence().to(device)
        optimizer = Adam(params=model.parameters(), lr=learningRate)
        criterion = nn.MSELoss()

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()

        # Save the initial hidden states (h0)
        fid = open("./result/initStates_from_best.txt", 'w')
        for idx in range(model.h0.shape[1]):
            for dim in range(model.h0.shape[2]):
                fid.write('%.8f\t' % model.h0[0][idx][dim].item())
            fid.write('\n')
        fid.close()


        # Save the model's output (y)
        Path("./result/output").mkdir(parents=True, exist_ok=True)
        for index, (nn_in, nn_label) in enumerate(db.data_loader):
            nn_in = nn_in.to(device)
            train_data = nn_in.permute(1, 0)  # (seqLength, batchsize)
            train_data = train_data.view(db.seqLenth, -1, model.output_dim)
            output = model.forward(train_data, nn_label)


            for idxBatch in range(output.shape[1]):
                print('Saving %d / %d' % (idxBatch+1,output.shape[1]))
                idxSeq = nn_label[idxBatch][0]

                plt.figure(figsize=(30, 10))
                plt.xlabel('time steps', fontsize=20)
                plt.ylabel('value', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                def draw(yi, color):
                    pData = yi.detach().cpu().numpy()
                    plt.plot(pData, color, linewidth=2.0)

                draw(output[:,idxBatch,0], 'r')
                draw(train_data[:, idxBatch, 0], 'b')
                plt.draw()
                plt.savefig('./result/output/predict_idxSeq_%d.png' % idxSeq)
                plt.close()
                fid = open("./result/output/predict_idxSeq_%d.txt" % idxSeq, 'w')
                for ff in range(output.shape[0]):
                    fid.write('%.8f\t%.8f\n' % (output[ff,idxBatch,0],train_data[ff, idxBatch, 0]))

                fid.close()