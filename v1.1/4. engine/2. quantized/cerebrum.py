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
## NAME: cerebrum.py (UCI chess engine)                                     ##
## AUTHOR: David Carteau, France, February 2025                             ##
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

NN_SIZE_L0 = 768
NN_SIZE_L1 = 256
NN_SIZE_L2 = 2


def nn_load(filepath):
    with open(filepath, "rt") as file:
        lines = file.readlines()
    #end with
    
    name = lines[0].replace("name=", "").strip()
    author = lines[1].replace("author=", "").strip()
    
    print(f'{name} by {author}')
    
    wr_factor = float(lines[2].replace("wr=", "").strip())
    mt_factor = float(lines[3].replace("mt=", "").strip())
    
    expected = int(lines[4].replace("parameters=", "").strip())
    
    lines = lines[5:]
    
    W0 = np.zeros((NN_SIZE_L1, NN_SIZE_L0), dtype=int)
    B0 = np.zeros(NN_SIZE_L1, dtype=int)
    
    W1 = np.zeros((NN_SIZE_L2, NN_SIZE_L1 * 2), dtype=int)
    B1 = np.zeros(NN_SIZE_L2, dtype=int)
    
    loaded = 0
    
    for array in [W0, B0, W1, B1]:
        if array.ndim == 1:
            for col in range(array.shape[0]):
                array[col] = int(lines[loaded])
                loaded += 1
            #end for
        else:
            for row in range(array.shape[0]):
                for col in range(array.shape[1]):
                    array[row, col] = int(lines[loaded])
                    loaded += 1
                #end for
            #end for
        #end if
    #end for
    
    print(f'Model: {loaded} parameters loaded ({expected} expected)')
    
    assert loaded == expected
    
    # PyTorch library transpose matrices to optimize dot products
    # for this educational engine example, we need here to transpose them back
    
    W0 = W0.T
    W1 = W1.T
    
    return W0, B0, W1, B1, wr_factor, mt_factor
#end def


def nn_evaluate(network, features_stm, features_opp):
    W0, B0, W1, B1, wr_factor, mt_factor = network
    
    acc_w = np.copy(B0)
    
    for p in features_stm:
        acc_w += W0[p]
    #end for
    
    acc_b = np.copy(B0)
    
    for p in features_opp:
        acc_b += W0[p]
    #end for
    
    L1 = np.concatenate((acc_w, acc_b), axis=0)
    L1 = np.clip(L1, 0, 127)
    
    L2 = np.matmul(L1, W1) + (B1 * 64)
    L2 //= 64
    
    L2 = L2.astype(float) / 64.0
    
    wr = L2[0]
    mt = L2[1]
    
    return (wr * wr_factor + mt * mt_factor)
#end def


def get_features(fenstring):
    fields = fenstring.strip().split()
    
    assert len(fields) >= 2
    
    fen, stm = fields[0], fields[1]
    
    assert stm in ["w", "b"]
    
    rows = fen.split("/")
    
    assert len(rows) == 8
    
    features_w = []
    features_b = []
    
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
            #end if
            
            index = "pnbrqk".find(char)
            
            if index != -1:
                feature_w = 64 * (2 * index + 1) + (square)
                feature_b = 64 * (2 * index + 0) + (square ^ 56)
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
            send("id name Cerebrum 1.1")
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
        
        if command.startswith("eval"):
            fenstring = board.fen()
            
            # transform fenstring in vectors of size 768 for each player
            # these vectors will be the input of the neural network for the evaluation
            # stm = side to move, opp = opponent
            
            features_stm, features_opp = get_features(fenstring)
            
            evaluation = nn_evaluate(network, features_stm, features_opp)
            evaluation = int(1000 * evaluation)
            
            send(f'info score cp {evaluation}')
            
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
                
                score = -nn_evaluate(network, features_stm, features_opp)
                
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
