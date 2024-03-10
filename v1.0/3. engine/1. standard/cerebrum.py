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
## NAME: cerebrum.py (UCI chess engine)                                     ##
## AUTHOR: David Carteau, France, March 2024                                ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## IMPORTANT !                                                              ##
## This chess engine is very weak: its only purpose is to illustrate how to ##
## use a neural network trained by the Cerebrum library                     ##
##############################################################################


import chess
import random

import numpy as np

NN_SIZE_L1 = 128
NN_SIZE_L2 = 32
NN_SIZE_L3 = 32
NN_SIZE_L4 = 2


def nn_load(filepath):
    with open(filepath, "rt") as file:
        lines = file.readlines()
    #end with
    
    name = lines[0].replace("name=", "").strip()
    author = lines[1].replace("author=", "").strip()
    
    print(f'{name} by {author}')
    
    wr_factor = float(lines[2].replace("wr=", "").strip())
    mt_factor = float(lines[3].replace("mt=", "").strip())
    
    lines = lines[4:]
    
    W0 = np.zeros((NN_SIZE_L1, 768))
    B0 = np.zeros(NN_SIZE_L1)
    
    W1 = np.zeros((NN_SIZE_L2, NN_SIZE_L1 * 2))
    B1 = np.zeros(NN_SIZE_L2)
    
    W2 = np.zeros((NN_SIZE_L3, NN_SIZE_L2))
    B2 = np.zeros(NN_SIZE_L3)
    
    W3 = np.zeros((NN_SIZE_L4, NN_SIZE_L3))
    B3 = np.zeros(NN_SIZE_L4)
    
    i = 0
    
    for array in [W0, B0, W1, B1, W2, B2, W3, B3]:
        if array.ndim == 1:
            for col in range(array.shape[0]):
                array[col] = float(lines[i])
                i += 1
            #end for
        else:
            for row in range(array.shape[0]):
                for col in range(array.shape[1]):
                    array[row, col] = float(lines[i])
                    i += 1
                #end for
            #end for
        #end if
    #end for
    
    # PyTorch library transpose matrices to optimize dot products
    # for this educational engine example, we need here to transpose them back
    
    W0 = W0.T
    W1 = W1.T
    W2 = W2.T
    W3 = W3.T
    
    print("Model has", i, "parameters")
    
    return W0, B0, W1, B1, W2, B2, W3, B3, wr_factor, mt_factor
#end def


def nn_eval(network, features_stm, features_opp):
    Q = 127.0 / 64.0
    
    W0, B0, W1, B1, W2, B2, W3, B3, wr_factor, mt_factor = network
    
    acc_w = np.copy(B0)
    
    for p in features_stm:
        acc_w += W0[p]
    #end for
    
    acc_b = np.copy(B0)
    
    for p in features_opp:
        acc_b += W0[p]
    #end for
    
    L1 = np.concatenate((acc_w, acc_b), axis=0)
    L1 = np.clip(L1, 0, Q)
    
    L2 = np.matmul(L1, W1) + B1
    L2 = np.clip(L2, 0, Q)
    
    L3 = np.matmul(L2, W2) + B2
    L3 = np.clip(L3, 0, Q)
    
    L4 = np.matmul(L3, W3) + B3
    L4 = np.clip(L4, -Q, Q)
    
    wr = L4[0]
    mt = L4[1]
    
    return (wr * wr_factor + mt * mt_factor)
#end def


def get_features(fenstring):
    features_w = []
    features_b = []
    
    whites = ["P", "N", "B", "R", "Q", "K"]
    blacks = ["p", "n", "b", "r", "q", "k"]
    
    fields = fenstring.strip().split()
    
    assert(len(fields) >= 2)
    
    fen, stm = fields[0], fields[1]
    
    assert(len(fen.split("/")) == 8)
    
    square = 56
    
    for row in fields[0].split("/"):
        for char in row:
            if char in whites:
                sq_w = square
                sq_b = square ^ 56
                index = 2 * whites.index(char)
                features_w.append(64 * (index + 0) + sq_w)
                features_b.append(64 * (index + 1) + sq_b)
            elif char in blacks:
                sq_w = square
                sq_b = square ^ 56
                index = 2 * blacks.index(char)
                features_w.append(64 * (index + 1) + sq_w)
                features_b.append(64 * (index + 0) + sq_b)
            elif char in "12345678":
                square += "12345678".find(char)
            #end if
            square += 1
        #end for
        
        square -= 16
    #end for
    
    assert(square == -8)
    
    features_w.sort()
    features_b.sort()
    
    assert(len(features_w) == len(features_b))
    
    for x in features_w + features_b:
        assert(x >= 0 and x < 768)
    #end for
    
    assert(stm in ["w", "b"])
    
    if stm == "w":
        return (features_w, features_b)
    else:
        return (features_b, features_w)
    #end if
#end def


def send(message):
    print(message, flush=True)
#end def


if __name__ == "__main__":
    random.seed(0)
    network = nn_load("./network.txt")
    
    while True:
        command = input().strip()
        
        if command.startswith("quit"):
            break
        #end if
        
        if command.startswith("ucinewgame"):
            random.seed(0)
            continue
        #end if
        
        if command.startswith("uci"):
            send("id name Cerebrum 1.0")
            send("id author David Carteau")
            send("option name Hash type spin default 0 min 0 max 0")
            send("uciok")
            continue
        #end if
        
        if command.startswith("isready"):
            send("readyok")
            continue
        #end if
        
        if command.startswith("position fen "):
            fenstring = " ".join(command.split()[2:8])
            board = chess.Board(fenstring)
        #end if
        
        if command.startswith("position startpos"):
            board = chess.Board()
        #end if
        
        if command.startswith("position ") and "moves " in command:
            command = command.split()
            idx = command.index("moves")
            moves = command[idx+1:]
            
            for move in moves:
                move = chess.Move.from_uci(move)
                board.push(move)
            #end for
            
            continue
        #end if
        
        if command.startswith("go"):
            bestmove, bestscore = None, None
            
            for move in board.legal_moves:
                board.push(move)
                
                fenstring = board.fen()
                
                board.pop()
                
                # transform fenstring in vectors of size 768 for each player
                # these vectors will be the input of the neural network for the evaluation
                # stm = side to move, opp = opponent
                
                features_stm, features_opp = get_features(fenstring)
                
                score = -nn_eval(network, features_stm, features_opp)
                
                if (bestscore is None) or (score > bestscore):
                    bestmove = move.uci()
                    bestscore = score
                #end if
            #end for
            
            bestscore = int(1000 * bestscore)
            
            send(f'info score cp {bestscore}')
            send(f'bestmove {bestmove}')
            
            continue
        #end if
    #end while
#end if
