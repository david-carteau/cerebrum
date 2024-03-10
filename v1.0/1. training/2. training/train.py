"""
The Cerebrum library and engine
Copyright (c) 2024, by David Carteau. All rights reserved.

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
## AUTHOR: David Carteau, France, March 2024                                ##
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
import numpy as np

from tqdm import tqdm
from scipy.sparse import csr_matrix

# random seed = date of Orion's first public release :-)
SEED = 21052014

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# set FORCE_CPU_DEVICE to False if you have a GPU (faster training)
FORCE_CPU_DEVICE = True

# USE_QUANTIZATION_ERROR to True to force the weights to converge towards quantized values
USE_QUANTIZATION_ERROR = False

# adjust if needed name, author, and balance between win ratio and material
NN_NAME = "Orion 1.0"
NN_AUTHOR = "David Carteau"
NN_WR = 0.5
NN_MT = 0.5

# adjust if needed network architecture
NN_SIZE_L1 = 128
NN_SIZE_L2 = 32
NN_SIZE_L3 = 32
NN_SIZE_L4 = 2

# Q and -Q (+/- 1,98) are the minimum and maximum values allowed for weights and biases
# this opens the possibility to use a quantized version of the network post-training
Q = 127/64

# do not modify (size of the input vectors)
HEIGHT = 768

# adjust if needed training hyper-parameters
DECAY = 1e-5
BATCH_SIZE = 1024 * 1024
MINI_BATCH_SIZE = 8 * 1024

# do not modify (structure of folders)
DATA_PATH = "./data"
MODELS_PATH = "./models"
NETWORKS_PATH = "./networks"
POSITIONS_PATH = "./positions"

# new learning rate scheduler (v1.0)
LR = [0.0100, 0.0050, 0.0010, 0.0005, 0.0001]

# old learning rate scheduler (v0.8 & v0.9)
#LR = [0.0050, 0.0045, 0.0040, 0.0035, 0.0030, 0.0025, 0.0020, 0.0015, 0.0010, 0.0005, 0.0001]

EPOCHS = len(LR)


# pretty formatting of float values
def format(n):
    n = "{:.05f}".format(n)
    
    if n == "-0.00000":
        n = "0.00000"
    #end if
    
    return n
#end def


# for each line of ./positions/positions-shuffled.txt:
# 1) convert the fenstring to two vectors (of size 768), one for current player, the other for its opponent
# 2) retrieve the win ratio and compute material value
def get_sample(line):
    sample_w = []
    sample_b = []
    
    mt = 0
    
    whites = ["P", "N", "B", "R", "Q", "K"]
    blacks = ["p", "n", "b", "r", "q", "k"]
    
    fields = line.strip().split()
    
    assert(len(fields) >= 3)
    assert(len(fields[0].split("/")) == 8)
    
    square = 56
    
    for row in fields[0].split("/"):
        for char in row:
            if char in whites:
                sq_w = square
                sq_b = square ^ 56
                index = 2 * whites.index(char)
                sample_w.append(64 * (index + 0) + sq_w)
                sample_b.append(64 * (index + 1) + sq_b)
                mt += {"P": 0.1, "N": 0.3, "B": 0.3, "R": 0.5, "Q": 0.9, "K": 0}[char]
            elif char in blacks:
                sq_w = square
                sq_b = square ^ 56
                index = 2 * blacks.index(char)
                sample_w.append(64 * (index + 1) + sq_w)
                sample_b.append(64 * (index + 0) + sq_b)
                mt -= {"p": 0.1, "n": 0.3, "b": 0.3, "r": 0.5, "q": 0.9, "k": 0}[char]
            elif char in "12345678":
                square += "12345678".find(char)
            #end if
            square += 1
        #end for
        
        square -= 16
    #end for
    
    assert(square == -8)
    
    sample_w.sort()
    sample_b.sort()
    
    assert(len(sample_w) == len(sample_b))
    
    for x in sample_w + sample_b:
        assert(x >= 0 and x < HEIGHT)
    #end for
    
    stm, wr = fields[1], float(fields[-1])
    
    assert(stm in ["w", "b"])
    assert(0.0 <= wr <= 1.0)
    
    if stm == "w":
        return (sample_w, sample_b, wr, mt)
    else:
        return (sample_b, sample_w, wr, -mt)
    #end if
#end def

# prepare a batch of samples:
# - get features (vectors for both players) and labels (win ratio & material)
# - convert features to sparse matrices to save disk space
def get_samples(lines):
    wr, mt = [], []
    
    indptr = [0]
    indices = []
    
    for line in lines:
        try:
            sample = get_sample(line)
        except:
            print("error", line)
            continue
        #end try
        
        indices += sample[0]
        indices += [f + HEIGHT for f in sample[1]]
        indptr += [indptr[-1] + len(sample[0]) + len(sample[1])]
        
        wr.append(sample[2])
        mt.append(sample[3])
    #end for
    
    data = [1] * len(indices)
    
    X = csr_matrix((data, indices, indptr), shape=(len(indptr)-1, HEIGHT * 2), dtype=np.int8)
    
    wr = np.array(wr)
    mt = np.array(mt)
    
    del data
    del indices
    del indptr
    
    return X, (wr, mt)
#end def


# load a batch from disk
def load_batch(batch):
    name = f'{DATA_PATH}/batch.{str(batch).zfill(2)}.pickle'
    
    if os.path.exists(name):
        with gzip.GzipFile(name, "rb") as file:
            return pickle.load(file)
        #end with
    #end if
    
    assert False
#end def


# convert sparse matrices back to normal arrays, if needed
def from_numpy(X, dtype=np.float32):
    if type(X) == csr_matrix:
        X = X.tocoo()
        X = torch.sparse_coo_tensor(np.vstack((X.row, X.col)), X.data, X.shape, dtype=torch.float32)
    else:
        X = X.astype(dtype)
        X = torch.from_numpy(X)
    #end if
    
    return X
#end def


# quantize a decimal value
def quantize(x):
    x = x * 64
    x = torch.round(x)
    x = x + 127
    x = torch.minimum(x, torch.full(x.shape, 254))
    x = (x - 127) / 64
    
    return x
#end def


# compute the difference between the value of a weight (or bias) and its nearest quantized version
def get_quantization_error(model):
    n_params = 0
    sum_errors = None
    
    for parameter in model.parameters():
        if parameter.dim() == 1:
            n_params += parameter.shape[0]
        else:
            n_params += parameter.shape[0] * parameter.shape[1]
        #end if
        
        if sum_errors is None:
            sum_errors = torch.abs(parameter - quantize(parameter)).sum()
        else:
            sum_errors = sum_errors + torch.abs(parameter - quantize(parameter)).sum()
        #end if
    #end for
    
    return sum_errors / n_params
#end def


# Orion v1.0 chess engine neural network model
# -> input: vanilla NNUE architecture
# -> output: 2 values predicted and combined (win ratio & material)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = torch.nn.Linear(HEIGHT, NN_SIZE_L1)
        self.fc2 = torch.nn.Linear(NN_SIZE_L1 * 2, NN_SIZE_L2)
        self.fc3 = torch.nn.Linear(NN_SIZE_L2, NN_SIZE_L3)
        self.fc4 = torch.nn.Linear(NN_SIZE_L3, NN_SIZE_L4)
    #end def
    
    def forward(self, x):
        x1, x2 = x
        
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        
        x = torch.cat((x1,x2), axis=1)
        x = torch.clamp(x, min=0, max=Q)
        
        x = self.fc2(x)
        x = torch.clamp(x, min=0, max=Q)
        
        x = self.fc3(x)
        x = torch.clamp(x, min=0, max=Q)
        
        x = self.fc4(x)
        x = torch.clamp(x, min=-Q, max=Q)
        
        cp = x[:,0]
        cp = torch.flatten(cp)
        
        mt = x[:,1]
        mt = torch.flatten(mt)
        
        return cp, mt
    #end def
#end class


# Trainer (regression problem, on two distinct values : win ratio & material)
class Regressor():
    def __init__(self):
        if FORCE_CPU_DEVICE or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda:0"
        #end if
        
        self.model = Net().to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        infos = ""
        
        if not torch.cuda.is_available():
            infos = "(no GPU available)"
        #end if
        
        if FORCE_CPU_DEVICE:
            infos = "(forced)"
        #end if
        
        print("PyTorch", torch.__version__)
        print("Using device:", self.device, infos)
        print()
        
        print("Model has", n_params, "parameters")
    #end def
    
    def fit(self, X, y):
        raise Exception("Unsupported method")
    #end def
    
    def partial_fit(self, X, y, lr=None, qr=None):
        self.model.train()
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=DECAY)
        
        X1, X2 = X
        y1, y2 = y
        
        # y1 = win ratio (renormalised between -1 and 1)
        # y2 = material (values clamped between -Q and Q)
        y1 = 2.0 * y1 - 1.0
        y2 = np.clip(y2, -Q, Q)
        
        losses = []
        q_errors = []
        
        mini_batches = [x for x in range(X1.shape[0] // MINI_BATCH_SIZE)]
        random.shuffle(mini_batches)
        
        for mini_batch in mini_batches:
            fr, to = MINI_BATCH_SIZE * mini_batch, MINI_BATCH_SIZE * (mini_batch + 1)
            
            X1_train = from_numpy(X1[fr:to,:]).to(self.device)
            X2_train = from_numpy(X2[fr:to,:]).to(self.device)
            
            y1_train = from_numpy(y1[fr:to]).to(self.device)
            y2_train = from_numpy(y2[fr:to]).to(self.device)
            
            X_train = (X1_train, X2_train)
            
            # Re-init gradients
            optimizer.zero_grad()
            
            # Forward
            y1_pred, y2_pred = self.model(X_train)
            
            # Compute loss
            loss = criterion(y1_pred, y1_train) + criterion(y2_pred, y2_train)
            
            # Quantization error
            if USE_QUANTIZATION_ERROR:
                q_error = get_quantization_error(self.model)
                q_errors.append(q_error.item())
                loss += (qr * q_error)
            #end if
            
            losses.append(loss.item())
            
            # Back-propagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Weights clipping
            with torch.no_grad():
                for param in self.model.parameters():
                    param.clamp_(-Q, Q)
                #end for
            #end with
        #end for
        
        # Print loss
        print("Training Loss:", format(np.mean(losses)), end=" ")
        
        if USE_QUANTIZATION_ERROR:
            print(" -  Q-Error:", format(np.mean(q_errors)))
        else:
            print()
        #end if
        
        return self
    #end def
    
    def predict(self, X):
        self.model.eval()
        
        with torch.no_grad():
            X1, X2 = X
            
            X1 = from_numpy(X1).to(self.device)
            X2 = from_numpy(X2).to(self.device)
            
            X = (X1, X2)
            Y = self.model(X)
        #end with
        
        return Y[0].cpu(), Y[1].cpu()
    #end def
#end class


# save one bunch of weights (or bias) in a row
def write_parameters(file, parameters, quantization):
    if quantization:
        values = [int(parameter.item() * 64) for parameter in quantize(parameters)]
    else:
        values = [parameter.item() for parameter in parameters]
    #end if
    
    values = [str(value) for value in values]
    file.write("\n".join(values) + "\n")
#end def


# save weights and biases
def torch_save_nn(model, epoch, quantization=False):
    print("Saving weights...")
    
    if quantization:
        path = f'{NETWORKS_PATH}/epoch-{epoch + 1}-q.txt'
    else:
        path = f'{NETWORKS_PATH}/epoch-{epoch + 1}.txt'
    #end if
    
    with open(path, "wt") as file:
        file.write(f'name={NN_NAME}\n')
        file.write(f'author={NN_AUTHOR}\n')
        file.write(f'wr={NN_WR}\n')
        file.write(f'mt={NN_MT}\n')
        
        for parameter in model.parameters():
            assert parameter.dim() in [1, 2]
            
            if parameter.dim() == 1:
                print("B:", len(parameter.data))

                write_parameters(file, parameter.data, quantization)
            else:
                n_rows = len(parameter.data)
                n_cols = len(parameter.data[0])
                
                print("W:", n_rows, "x", n_cols)
                
                for row in parameter.data:
                    write_parameters(file, row, quantization)
                #end for
            #end if
        #end for
    #end with
    
    print("Weights saved !")
    print()
#end def


# main function
def main():
    print("Cerebrum library v1.0 (2024)")
    print("https://github.com/david-carteau/cerebrum")
    print("Neural network trainer for the UCI chess engine Orion")
    print()
    
    # creation of folders structure
    
    for path in [DATA_PATH, MODELS_PATH, NETWORKS_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)
        #end if
    #end for
    
    # preparation of batches
    
    n_batch = 0
    n_positions = 0
    
    with open(f'{POSITIONS_PATH}/positions-shuffled.txt') as file:
        lines = []
        
        for line in tqdm(file):
            n_positions += 1
            
            line = line.strip()
            lines.append(line)
            
            if len(lines) == BATCH_SIZE:
                n_batch += 1
                
                name = f'{DATA_PATH}/batch.{str(n_batch).zfill(2)}.pickle'
                
                if not os.path.exists(name):
                    X, y = get_samples(lines)
                    
                    X1 = X[:,:HEIGHT]
                    X2 = X[:,HEIGHT:]

                    with gzip.GzipFile(name, "wb", compresslevel=6) as file:
                        pickle.dump(((X1, X2), y), file)
                    #end with
                #end if
                
                lines = []
            #end if
        #end for
        
        if len(lines):
            n_batch += 1
            
            name = f'{DATA_PATH}/batch.{str(n_batch).zfill(2)}.pickle'
            
            if not os.path.exists(name):
                X, y = get_samples(lines)
                
                X1 = X[:,:HEIGHT]
                X2 = X[:,HEIGHT:]
                
                with gzip.GzipFile(name, "wb", compresslevel=6) as file:
                    pickle.dump(((X1, X2), y), file)
                #end with
            #end if
            
            lines = []
        #end if
        
        lines = None
    #end with
    
    print()
    
    batches = [x + 1 for x in range(n_batch)]
    
    n_training = (n_batch - 1) * BATCH_SIZE
    n_validation = n_positions - n_training
    
    print(f'Total: {n_positions} positions')
    print(f'Training: {n_training} positions ({n_batch - 1} batches of {BATCH_SIZE} positions)')
    print(f'Validation: {n_validation} positions (one batch)')
    print()
    
    # let's go !
    
    reg = Regressor()
    
    print()
    
    X_test, y_test = load_batch(batches[-1])
    
    batches = batches[:-1]
    
    for epoch, rate in enumerate(LR):
        if USE_QUANTIZATION_ERROR:
            qr = (epoch + 1) / EPOCHS
        else:
            qr = None
        #end if
        
        random.shuffle(batches)
        
        # training
        
        for i, batch in enumerate(batches):
            print("Epoch", epoch + 1, "/", EPOCHS, " - ", end=" ")
            print("Batch", i + 1, "/", len(batches), "(n°" + str(batch) + ")", " - ", end=" ")
            print("lr =", format(rate), end=" ")
            
            if USE_QUANTIZATION_ERROR:
                print(" -  qr =", format(qr))
            else:
                print()
            #end if
            
            X_train, y_train = load_batch(batch)
            reg.partial_fit(X_train, y_train, lr=rate, qr=qr)
            
            print()
        #end for
        
        # evaluation (with the validation dataset)
        
        y1_true, y2_true = y_test
        
        # y1_true = win ratio, y2_true = material (see also lines 363 & 364)
        y1_true = 2.0 * y1_true - 1.0
        y2_true = np.clip(y2_true, -Q, Q)
        
        y1_pred, y2_pred = reg.predict(X_test)
        
        se_wr = (y1_true - y1_pred.numpy())**2
        se_mt = (y2_true - y2_pred.numpy())**2
        
        mse = format(np.mean(se_wr + se_mt))
        
        mse_wr = format(np.mean(se_wr))
        mse_mt = format(np.mean(se_mt))
        
        print("MSE:", mse, "=", "MSE(WR):", mse_wr, "+", "MSE(MT):", mse_mt)
        print()
        
        # weights saving
        
        with open(f'{MODELS_PATH}/epoch-{epoch + 1}_(mse={mse}).pickle', "wb") as file:
            pickle.dump(reg, file)
        #end with
        
        torch_save_nn(reg.model, epoch)
        torch_save_nn(reg.model, epoch, quantization=True)
    #end for
#end with

if __name__ == "__main__":
    main()
#end if
