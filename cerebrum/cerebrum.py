"""
The Cerebrum library
Copyright (c) 2020, by David Carteau. All rights reserved.

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
## NAME: cerebrum.py (from Orion UCI chess engine)                          ##
## AUTHOR: David Carteau, France, December 2020                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

# The library is composed of two parts :
# - cerebrum.py : Python program which aims at training neural networks
#   used in the Orion UCI chess engine
# - cerebrum.h and cerebrum.c : inference code which can be embedded in a
#   chess engine

"""
Preparation:
1) download and install Python 3.8.6
2) install pytorch 1.7.0, following instructions given on this page:
   https://pytorch.org/get-started/locally/
   or directly by launching the follwoing command (under the console):
   "pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html"
3) install scipy 1.5.4 using :
   "pip install scipy"
4) downgrade numpy to 1.19.3 using :
   "pip install numpy==1.19.3"
   ==> see : https://github.com/numpy/numpy/issues/17726
5) prepare a text file, where each line follows the format: fen;eval_w;eval_b
   where 'fen' is the first part of a fenstring (i.e. without stm, castling
   rights, etc.), and 'eval_w' is the evaluation in pawns (e.g.: 0.42) from
   the white perspective (idem for 'eval_b')
6) shuffle all the lines of the text file : this is really important !
7) put this script and the prepared text file into a directory, which must
   also contains two empty sub-folders called "data" and "networks"

Usage:
1) specify the name of your text file by modifying the POSITIONS_FILE variable
   below and modify other parameters if needed (e.g.: LEARNING_RATE)
2) launch this script (no parameter)
3) first launch will index your text file, and then stop
4) relaunch the script to start the training
5) first "epoch" (first pass on the whole text file) will be slow, because it
   will transform text data into binary format (gzipped to save disk space),
   following epochs will be greatly faster especially if you have a NVIDIA gpu

Notes:
1) training will save a network file at the end of each epoch (text format)
2) this raw network will be converted in binary format by the inference code
   (see cerebrum.c)
3) training is designed to be reproducible (randomness is driven by the SEED
   variable)
4) you can interrupt training as you want, and resume it later: progress is
   stored at the end of each epoch (cf. MODEL variable), but you will then lose
   reproducibility
5) during training, two metrics will be shown: L1 which is the average error
   with the targeted values, and L2 which is the sum of squarred errors
6) training can be manually stopped when these two values do not significantly
   vary anymore
"""



##############################################################################
## CONSTANTS                                                                ##
##############################################################################

POSITIONS_FILE = "positions.txt"
LEARNING_RATE = 0.03
MAX_EPOCHS = 1000
SEED = 0

# WIDTH is equivalent to NN_SIZE in cerebrum.h
# set it to 256 to have an architecture similar to Stockfish's one
WIDTH = 128
RELU_MIN = 0
RELU_MAX = 1

HEIGHT = 40960 # do not modify
MINI_BATCH_SIZE = 256*1024
MODEL = "networks/state_dict_model.pt"



##############################################################################
## IMPORTS                                                                  ##
##############################################################################

import gzip
import math
import torch
import pickle
import random
import numpy as np

from datetime import datetime
from scipy.sparse import csr_matrix



##############################################################################
## FUNCTIONS                                                                ##
##############################################################################

def get_sample(line):
	sample_w = []
	sample_b = []
	
	whites = ["P", "N", "B", "R", "Q"]
	blacks = ["p", "n", "b", "r", "q"]
	
	king_w = None
	king_b = None
	
	fields = line.split(";")
	
	assert(len(fields) == 3)
	assert(len(fields[0].split("/")) == 8)
	
	square = 56
	for row in fields[0].split("/"):
		for char in row:
			if char == "K":
				king_w = square
			elif char == "k":
				king_b = square
			elif char in "12345678":
				square += "12345678".find(char)
			#end if
			square += 1
		#end for
		
		square -= 16
	#end for
	
	assert(square == -8)
	assert(king_w is not None)
	assert(king_b is not None)
	
	king_b ^= 63
	
	square = 56
	for row in fields[0].split("/"):
		for char in row:
			if char in whites:
				sq_w = square
				sq_b = square ^ 63
				index = 2 * whites.index(char)
				sample_w.append( (king_w) * 640 + (index + 0) * 64 + (sq_w) )
				sample_b.append( (king_b) * 640 + (index + 1) * 64 + (sq_b) )
			elif char in blacks:
				sq_w = square
				sq_b = square ^ 63
				index = 2 * blacks.index(char)
				sample_w.append( (king_w) * 640 + (index + 1) * 64 + (sq_w) )
				sample_b.append( (king_b) * 640 + (index + 0) * 64 + (sq_b) )
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
	
	eval_w = float(fields[1])
	eval_b = float(fields[2])
	
	return (sample_w, sample_b, eval_w, eval_b)
#end def

def load_chunk(chunk):
	#print("Loading chunk...", chunk)
	
	assert(INDEX is not None)
	
	with open(POSITIONS_FILE, "r") as file:
		lines = []
		file.seek(INDEX[chunk], 0)
		for line in file:
			lines.append(line)
			if len(lines) >= MINI_BATCH_SIZE // 2:
				break
			#end if
		#end for
	#end with
	
	#print("Transforming data...")
	
	y = []
	indptr = [0]
	indices = []
	
	for line in lines:
		try:
			sample = get_sample(line)
		except:
			print("Error while parsing:", line)
			continue
		#end try
		
		len_features = len(sample[0]) + len(sample[1])
		
		if len_features == 0:
			continue
		#end if
		
		indices += sample[0]
		indices += [f + HEIGHT for f in sample[1]]
		indptr += [indptr[-1] + len_features]
		
		indices += sample[1]
		indices += [f + HEIGHT for f in sample[0]]
		indptr += [indptr[-1] + len_features]
		
		y.append(sample[2])
		y.append(sample[3])
	#end for
	
	data = [1] * len(indices)
	
	X = csr_matrix((data, indices, indptr), shape=(len(indptr)-1, HEIGHT * 2), dtype=np.int8)
	y = np.array(y)
	
	assert(np.min(X) == 0 and np.max(X) == 1)
	
	#print("Data transformed")
	
	return X, y
#end def

def prepare_index():
	print("Indexing data...")
	
	index = []
	lines = 0
	cursor = 0
	
	with open(POSITIONS_FILE, "r") as file:
		for line in file:
			if (lines % (MINI_BATCH_SIZE // 2)) == 0:
				index.append(cursor)
			#end if
			
			lines += 1
			cursor += len(line) + 1
		#end for
	#end with
	
	with open("data/positions.index", "wb") as file:
		pickle.dump(index, file)
	#end with
	
	print("Data indexed")
#end def

def prepare_data():
	try:
		with open("data/positions.index", "rb") as file:
			return pickle.load(file)
		#end with
	except:
		prepare_index()
		exit()
	#end
#end def

def torch_get_weights(tensor_row):
	return "\n".join([str(tensor.item()) for tensor in tensor_row]) + "\n"
#end def

def torch_save_nn(model, filename):
	print("Saving weights...")
	with open(filename, "w") as file:
		for parameter in model.parameters():
			if parameter.dim() == 1:
				print("B:", len(parameter.data), "x", 1)
				row = parameter.data
				file.write(torch_get_weights(row))
			elif parameter.dim() == 2:
				print("W:", len(parameter.data), "x", len(parameter.data[0]))
				for row in parameter.data:
					file.write(torch_get_weights(row))
				#end for
			else:
				assert(0)
			#end if
		#end for
	#end with
	print("Weights saved")
#end def

def convert_sparse_matrix_to_sparse_tensor(X):
	coo = X.tocoo()
	indices = np.vstack((coo.row, coo.col))
	return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float32)
#end def

""" Model definition """

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.fc1 = torch.nn.Linear(HEIGHT, WIDTH)
		self.fc2 = torch.nn.Linear(WIDTH * 2, 32)
		self.fc3 = torch.nn.Linear(32, 32)
		self.fc4 = torch.nn.Linear(32, 1)
	#end def
	
	def forward(self, x):
		x1, x2 = x
		x1 = self.fc1(x1)
		x2 = self.fc1(x2)
		x = torch.cat((x1,x2), axis=1)
		
		x = torch.clamp(x, min=RELU_MIN, max=RELU_MAX)
		
		x = self.fc2(x)
		x = torch.clamp(x, min=RELU_MIN, max=RELU_MAX)
		
		x = self.fc3(x)
		x = torch.clamp(x, min=RELU_MIN, max=RELU_MAX)
		
		x = self.fc4(x)
		
		return x
	#end def
#end class



##############################################################################
## PROGRAM                                                                  ##
##############################################################################

index = prepare_data()

""" Training """

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

try:
	model.load_state_dict(torch.load(MODEL))
	print("Previous model loaded !")
except:
	print("No previous model found...")
#end try

# the optimization algorithm can be easily changed
# please refer to https://pytorch.org/docs/stable/optim.html for more info
optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

MAX_CHUNKS = len(index) - 1

loss_1_fn = torch.nn.L1Loss()
loss_2_fn = torch.nn.MSELoss(reduction="sum")

print("Start training using device:", device)

for epoch in range(MAX_EPOCHS):
	batch = 0
	
	for chunk in random.sample(range(MAX_CHUNKS), MAX_CHUNKS):
		batch += 1
		
		print("epoch", epoch + 1, "batch", batch, "/", MAX_CHUNKS, "(" + str(chunk) + ") =>", datetime.now())
		
		chunk_file = "data/" + str(chunk) + ".pickle"
		
		try:
			with gzip.GzipFile(chunk_file, "rb") as file:
				(X1, X2, y) = pickle.load(file)
			#end with
		except:
			X, y = load_chunk(chunk)
			assert(X is not None and y is not None)
			
			X1 = convert_sparse_matrix_to_sparse_tensor(X[:,:HEIGHT])
			X2 = convert_sparse_matrix_to_sparse_tensor(X[:,HEIGHT:])
			y = torch.from_numpy(y.reshape(y.shape[0], 1).astype(np.float32))
			
			with gzip.GzipFile(chunk_file, "wb") as file:
				pickle.dump((X1, X2, y), file)
			#end with
		#end try
		
		X1 = X1.to(device)
		X2 = X2.to(device)
		X = (X1, X2)
		
		y = y.to(device)
		
		# gradient descent
		
		optimizer.zero_grad()
		
		Y = model(X)
		
		loss_1 = loss_1_fn(Y, y)
		loss_2 = loss_2_fn(Y, y)
		loss_2.backward()
		
		optimizer.step()
		
		# loss
		
		L1 = round(loss_1.item(), 2)
		L2 = round(loss_2.item(), 2)
		
		print("Loss L1:", L1, "- Loss L2:", L2)
		
		del X
		del Y
		del y
	#end for
	
	torch.save(model.state_dict(), MODEL)
	
	model.to("cpu")
	network_name = "networks/network-" + str(WIDTH) + "-epoch-" + str(epoch + 1) + ".txt"
	torch_save_nn(model, network_name)
	model.to(device)
#end for

network_name = "networks/network-" + str(WIDTH) + ".txt"
torch_save_nn(model, network_name)
