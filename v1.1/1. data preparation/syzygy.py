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
## NAME: syzygy.py                                                          ##
## AUTHOR: David Carteau, France, February 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Replace win ratio with actual Syzygy evals (3, 4, 5 and 6 pieces tables) ##
##############################################################################

import os
import shutil
import chess.syzygy

from tqdm import tqdm


def syzygy():
    test_positions = {
        3: "4R3/5K2/3k4/8/8/8/8/8 b - - 0 1",
        4: "4R3/5r2/3k1K2/8/8/8/8/8 w - - 0 1",
        5: "8/1p6/1P6/8/P7/3k4/5K2/8 b - - 0 1",
        6: "3b4/3P4/7k/p1K5/8/8/6P1/8 w - - 0 1"
    }
    
    for pop in range(3, 7):
        print(pop, "pieces")
        
        if pop < 6:
            tablebase = "./syzygy/3-4-5"
        else:
            tablebase = "./syzygy/6-pieces"
        #end if
        
        with chess.syzygy.open_tablebase(tablebase) as syzygy:
            board = chess.Board(test_positions[pop])
            wdl = syzygy.get_wdl(board)
            
            if wdl is None:
                print("Tablebase not available !")
                continue
            #end if
        #end with
        
        path = f'./set/popcount-{pop}.txt'
        save = f'./set/popcount-{pop}-original.txt'
        
        assert not os.path.exists(save)
        
        shutil.copy(path, save)
        
        errors = 0
        
        with chess.syzygy.open_tablebase(tablebase) as syzygy:
            with open(save, 'rt') as i_file, open(path, 'wt') as o_file:
                for line in tqdm(i_file):
                    pos, stm, pop, cnt, wr = line.strip().split()
                    
                    board = chess.Board(f'{pos} {stm} - - 0 1')
                    
                    wdl = syzygy.get_wdl(board)
                    
                    if wdl is None:
                        errors += 1
                    else:
                        wr = {-2: "0.0", -1: "0.0", 0: "0.5", 1: "1.0", 2: "1.0"}[wdl]
                    #end if
                    
                    o_file.write(f'{pos} {stm} {pop} {cnt} {wr}\n')
                #end for
            #end with
        #end with
        
        print(errors, "error(s)")
        print()
    #end for
#end def


if __name__ == "__main__":
    syzygy()
#end if
