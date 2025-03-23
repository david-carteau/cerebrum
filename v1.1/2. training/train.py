"""
The Cerebrum library and engine
Copyright (c) 2025, by David Carteau. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

##############################################################################
## NAME: train.py                                                           ##
## AUTHOR: David Carteau, France, March 2025                                ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Train the neural network, store result in ./networks folder              ##
##############################################################################

import os
import gzip
import torch
import pickle
import random

from tqdm import tqdm


# random seed (adjust if needed)
# 21.05.2014 = date of Orion's first public release :-)
SEED = 21052014

# whether to force training on CPU only (adjust if needed)
# set to True for better reproducibilty
# set to False if you have a GPU (faster training)
FORCE_CPU_DEVICE = False

# name, author, and balance between win ratio and material (adjust if needed)
NN_NAME = "Cerebrum 1.1"
NN_AUTHOR = "David Carteau"
NN_WR = 0.5
NN_MT = 0.5

# network architecture (important note)

# Make sure to use:
# - 768 for the first layer
# - None, or multiples of 32, for the hidden layer(s)
# - 2 for the last layer

# For the hidden layer(s), multiples of 32 are required ONLY if you use:
# - the provided C inference code
# - with the AVX and/or FMA set(s) of instructions

# network architecture (adjust if needed)
NN_SIZE_L0 = 768
NN_SIZE_L1 = 128
NN_SIZE_L2 = 32
NN_SIZE_L3 = 2
NN_SIZE_L4 = None

# network architecture (example with 0 hidden layer)
# NN_SIZE_L0 = 768
# NN_SIZE_L1 = 256
# NN_SIZE_L2 = 2
# NN_SIZE_L3 = None
# NN_SIZE_L4 = None

# network architecture (example with 1 hidden layer)
# NN_SIZE_L0 = 768
# NN_SIZE_L1 = 128
# NN_SIZE_L2 = 32
# NN_SIZE_L3 = 2
# NN_SIZE_L4 = None

# network architecture (example with 2 hidden layers, used in Orion v1.0 engine)
# NN_SIZE_L0 = 768
# NN_SIZE_L1 = 128
# NN_SIZE_L2 = 32
# NN_SIZE_L3 = 32
# NN_SIZE_L4 = 2

# check network architecture
if NN_SIZE_L4 is not None:
    assert NN_SIZE_L3 is not None
#end if

# learning rate scheduler (adjust if needed)
LR = [0.0050, 0.0045, 0.0040, 0.0035, 0.0030, 0.0025, 0.0020, 0.0015, 0.0010, 0.0005, 0.0001]

# batch size (adjust if needed)
# note that if you change this value, you'll need to remove the ./cache folder
# and/or delete all of its content for it to take effect
BATCH_SIZE = 64 * 1024

# weight decay, i.e. try to reduce the magnitude of weights and biases (adjust if needed)
DECAY = 1e-5

# option to convert material value to win ratio (adjust if needed)
CONVERT_CP_TO_WR = False

# option to help the network converge towards quantized weights and biases (adjust if needed)
# set to True if you intend to use quantized networks (better speed)
# set to False if you intend to use standard networks (better accuracy)
USE_QUANTIZATION_ERROR = True

# number of epochs (do not modify: modify learning rate scheduler instead)
N_EPOCHS = len(LR)

# Q value (do not modify)
# Q and -Q (+/- 1.98) are the minimum and maximum values allowed for weights and biases
# this opens the possibility to use a quantized version of the network (post-training)
Q = 127 / 64

# structure of folders (do not modify)
CACHE_PATH = "./cache"
MODELS_PATH = "./models"
NETWORKS_PATH = "./networks"
POSITIONS_PATH = "./positions"

# device used for the training (do not modify)
if torch.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
#end if

if FORCE_CPU_DEVICE:
    DEVICE = "cpu"
#end if


# load dataset and prepare/cache batches if necessary
class Dataset():
    def __init__(self):
        dataset = f'{POSITIONS_PATH}/positions-shuffled.txt'
        
        lines = []
        batch = 0
        
        print("Reading dataset...")
        
        with open(dataset, 'rt') as file:
            for line in tqdm(file):
                lines.append(line)
                
                if len(lines) == BATCH_SIZE:
                    path = f'{CACHE_PATH}/batch.{batch}.pickle'
                    
                    if not os.path.exists(path):
                        self.save_batch(batch, lines)
                    #end if
                    
                    lines = []
                    batch += 1
                #end if
            #end for
        #end with
        
        lines = None
        
        print("Done !")
        print()
        
        self.batches = [i for i in range(batch)]
    #end def
    
    def __len__(self):
        return len(self.batches)
    #end def
    
    def __iter__(self):
        self.batch = -1
        return self
    #end def
    
    def __next__(self):
        self.batch += 1
        
        if self.batch == len(self.batches):
            random.shuffle(self.batches)
            raise StopIteration
        #end if
        
        batch = self.batches[self.batch]
        
        return self.load_batch(batch)
    #end def
    
    # for each line of ./positions/positions-shuffled.txt:
    # 1) convert the fenstring to two vectors (of size 768), one for current player, the other for its opponent
    # 2) retrieve the win ratio and compute material value
    def get_sample(self, line):
        features_w = []
        features_b = []
        
        fen, stm, pop, cnt, wr = line.strip().split()
        
        wr = float(wr)
        mt = 0
        
        rows = fen.split("/")
        
        assert len(rows) == 8
        assert stm in ["w", "b"]
        assert 0.0 <= wr <= 1.0
        
        square = 0
        
        for row in rows[::-1]:
            for char in row:
                index = "12345678".find(char)
                
                if index != -1:
                    square += index + 1
                    continue
                #end if
                
                index = "PNBRQK".find(char)
                
                if index != -1:
                    feature_w = 64 * (2 * index + 0) + (square)
                    feature_b = 64 * (2 * index + 1) + (square ^ 56)
                    mt += [100, 300, 300, 500, 900, 0][index]
                #end if
                
                index = "pnbrqk".find(char)
                
                if index != -1:
                    feature_w = 64 * (2 * index + 1) + (square)
                    feature_b = 64 * (2 * index + 0) + (square ^ 56)
                    mt -= [100, 300, 300, 500, 900, 0][index]
                #end if
                
                features_w.append(feature_w)
                features_b.append(feature_b)
                
                square += 1
            #end for
        #end for
        
        assert square == 64
        
        features_w.sort()
        features_b.sort()
        
        if stm == "w":
            return (features_w, features_b, wr, mt)
        else:
            return (features_b, features_w, wr, -mt)
        #end if
    #end def
    
    # prepare a batch of samples:
    # - get features (vectors for both players) and labels (win ratio & material)
    # - convert features to sparse tensors to save memory
    def get_samples(self, lines):
        n_samples = len(lines)
        n_features = NN_SIZE_L0
        
        rows = []
        
        X1, X2 = [], []
        wr, mt = [], []
        
        for i, line in enumerate(lines):
            try:
                sample = self.get_sample(line)
            except Exception as e:
                print("error", line)
                print(e)
                input("Press a key...")
                continue
            #end try
            
            rows += [i] * len(sample[0])
            
            X1 += sample[0]
            X2 += sample[1]
            
            wr.append(sample[2])
            mt.append(sample[3])
        #end for
        
        data = [1.0] * len(X1)
        shape = (n_samples, n_features)
        
        X1 = torch.sparse_coo_tensor((rows, X1), data, shape, dtype=torch.float32)
        X2 = torch.sparse_coo_tensor((rows, X2), data, shape, dtype=torch.float32)
        
        wr = torch.tensor(wr, dtype=torch.float32)
        mt = torch.tensor(mt, dtype=torch.float32)
        
        return (X1, X2), (wr, mt)
    #end def
    
    # load a batch from disk
    def load_batch(self, batch):
        name = f'{CACHE_PATH}/batch.{batch}.pickle'
        
        with gzip.GzipFile(name, "rb") as file:
            return pickle.load(file)
        #end with
    #end def
    
    # save a batch to disk
    def save_batch(self, batch, lines):
        name = f'{CACHE_PATH}/batch.{batch}.pickle'
        
        X, y = self.get_samples(lines)
        
        with gzip.GzipFile(name, "wb", compresslevel=6) as file:
            pickle.dump((X, y), file)
        #end with
    #end def
#end class


# chess engine neural network model
# -> input : 768 features (6 types of piece x 2 colors x 64 squares = 768)
# -> output: 2 values (win ratio & material)
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.fc1 = torch.nn.Linear(NN_SIZE_L0, NN_SIZE_L1)
        self.fc2 = torch.nn.Linear(NN_SIZE_L1 * 2, NN_SIZE_L2)
        
        if NN_SIZE_L3 is not None:
            self.fc3 = torch.nn.Linear(NN_SIZE_L2, NN_SIZE_L3)
        #end if
        
        if NN_SIZE_L4 is not None:
            self.fc4 = torch.nn.Linear(NN_SIZE_L3, NN_SIZE_L4)
        #end if
    #end def
    
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    #end def
    
    def forward(self, x):
        x1, x2 = x
        
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        
        x = torch.cat((x1,x2), axis=1)
        
        x = torch.clamp(x, min=0, max=Q)
        x = self.fc2(x)
        
        if NN_SIZE_L3 is not None:
            x = torch.clamp(x, min=0, max=Q)
            x = self.fc3(x)
        #end if
        
        if NN_SIZE_L4 is not None:
            x = torch.clamp(x, min=0, max=Q)
            x = self.fc4(x)
        #end if
        
        wr = x[:,0]
        wr = torch.flatten(wr)
        
        mt = x[:,1]
        mt = torch.flatten(mt)
        
        return wr, mt
    #end def
    
    # quantization of decimal values
    def quantize(self, x):
        q_x = x.clone()
        
        q_x = q_x * 64
        q_x = torch.round(q_x)
        q_x = torch.clamp(q_x, min=-127, max=127)
        q_x = q_x / 64
        
        return q_x
    #end def
    
    # quantization mean squared error
    def quantization_error(self):
        sum_errors = 0
        
        for parameter in self.parameters():
            if parameter.requires_grad:
                error = (parameter - self.quantize(parameter))**2
                sum_errors += error.sum()
            #end if
        #end for
        
        return sum_errors / self.n_params()
    #end def
    
    # save a bunch of weights (or biases) in a row
    def save_parameters(self, file, parameters, quantization):
        if quantization:
            values = [int(parameter.item() * 64) for parameter in self.quantize(parameters)]
        else:
            values = [parameter.item() for parameter in parameters]
        #end if
        
        values = [str(value) for value in values]
        file.write("\n".join(values) + "\n")
    #end def
    
    # save the entire network (weights and biases)
    def save_network(self, epoch, quantization=False, verbose=False):
        if verbose:
            print("Saving quantized" if quantization else "Saving", "network...")
        #end if
        
        if quantization:
            path = f'{NETWORKS_PATH}/epoch-{epoch + 1}-q.txt'
        else:
            path = f'{NETWORKS_PATH}/epoch-{epoch + 1}.txt'
        #end if
        
        with open(path, "wt") as file:
            file.write(f'name={NN_NAME}\n')
            file.write(f'author={NN_AUTHOR}\n')
            file.write(f'parameters={self.n_params()}\n')
            file.write(f'wr={NN_WR}\n')
            file.write(f'mt={NN_MT}\n')
            
            for parameter in self.parameters():
                assert parameter.dim() in [1, 2]
                
                if parameter.dim() == 1:
                    if verbose:
                        print("B:", len(parameter.data))
                    #end if
                    
                    self.save_parameters(file, parameter.data, quantization)
                else:
                    n_rows = len(parameter.data)
                    n_cols = len(parameter.data[0])
                    
                    if verbose:
                        print("W:", n_rows, "x", n_cols)
                    #end if
                    
                    for row in parameter.data:
                        self.save_parameters(file, row, quantization)
                    #end for
                #end if
            #end for
        #end with
        
        if verbose:
            print("Quantized network" if quantization else "Network", "saved !")
            print()
        #end if
    #end def
    
    def save(self, epoch, loss):
        model_path = f'{MODELS_PATH}/epoch-{epoch + 1}-{format(loss)}.pt'
        torch.save(self, model_path)
        
        self.save_network(epoch)
        self.save_network(epoch, quantization=True)
    #end def
#end class


# trainer (regression problem, on two distinct values : win ratio & material)
class Trainer():
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = Network().to(DEVICE)
        
        print("PyTorch", torch.__version__)
        print("Using device:", DEVICE, "(forced)" if FORCE_CPU_DEVICE else "")
        print()
        
        print("Model has", self.model.n_params(), "parameters")
        print()
        
        lr_lambda = lambda epoch: LR[epoch] if epoch < len(LR) else 0
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0, weight_decay=DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    #end def
    
    def train_one_batch(self, X, y):
        X1, X2 = X
        wr, mt = y
        
        # remormalise win ratio between -1.0 and 1.0
        wr = 2.0 * wr - 1.0
        
        # adjust material value, 2 options available:
        # - convert material value to win ratio
        # - rescale material value in order to have something comparable in scale with win ratio
        
        # the second option is the default for Orion, because it helps the engine to play endgames better
        # it allows it to make a significant difference (in terms of evaluation) between largely winning positions
        # e.g. KRRvK and KRvK positions: in both cases, stronger side will win, but more easily/quickly with one rook ahead
        
        if CONVERT_CP_TO_WR:
            # inspired by https://www.chessprogramming.org/Pawn_Advantage,_Win_Percentage,_and_Elo
            mt = 1 / (1 + 10**(-mt/400))
            mt = 2.0 * mt - 1.0
        else:
            # idea: one queen ahead --> mt = +0.9 --> close to / similar to a win ratio of +1.0
            mt /= 1000.0
        #end if
        
        # move data to the training device (i.e. cpu or gpu)
        X1_true, X2_true = X1.to(DEVICE), X2.to(DEVICE)
        wr_true, mt_true = wr.to(DEVICE), mt.to(DEVICE)
        
        X_true = (X1_true, X2_true)
        
        # re-init gradients
        self.optimizer.zero_grad()
        
        # predict output (i.e. forward pass)
        wr_pred, mt_pred = self.model(X_true)
        
        # compute loss
        loss = self.criterion(wr_pred, wr_true) + self.criterion(mt_pred, mt_true)
        
        # optionally, add quantization error to the loss
        if USE_QUANTIZATION_ERROR:
            loss += self.model.quantization_error()
        #end if
        
        # back-propagate the loss
        loss.backward()
        
        # update model weights
        self.optimizer.step()
        
        # clamp model weights (in the perspective of quantization)
        with torch.no_grad():
            for param in self.model.parameters():
                param.clamp_(-Q, Q)
            #end for
        #end with
        
        return loss.item()
    #end def
    
    def train_one_epoch(self, epoch):
        loss = 0
        losses = []
        
        # iterate through the dataset
        for X, y in (pbar := tqdm(self.dataset)):
            loss = self.train_one_batch(X, y)
            losses.append(loss)
            
            loss = losses[-32:]
            loss = sum(loss) / len(loss)
            
            last = self.scheduler.get_last_lr()[0]
            
            pbar.set_postfix({"LR:": format(last), "Loss:": format(loss)})
        #end for
        
        # update learning rate
        self.scheduler.step()
        
        # save model
        self.model.save(epoch, loss)
    #end def
#end class


# pretty formatting of float values
def format(n):
    n = f"{n:.05f}"
    
    if n == "-0.00000":
        n = "0.00000"
    #end if
    
    return n
#end def


# main function
def main():
    print("Cerebrum library v1.1 (2025)")
    print("https://github.com/david-carteau/cerebrum")
    print("NNUE Neural network trainer for chess engines")
    print()
    
    # set seed for reproducibilty
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # creation of folders structure
    for path in [CACHE_PATH, MODELS_PATH, NETWORKS_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)
        #end if
    #end for
    
    # let's go !
    dataset = Dataset()
    trainer = Trainer(dataset)
    
    for epoch in range(N_EPOCHS):
        print("Epoch:", epoch + 1, "/", N_EPOCHS)
        
        trainer.train_one_epoch(epoch)
        
        print()
    #end for
    
    print("Training complete !")
    print()
#end with


if __name__ == "__main__":
    main()
#end if