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
## NAME: syzygy.py                                                          ##
## AUTHOR: David Carteau, France, March 2024                                ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Replace win ratio with actual Syzygy evals (3, 4, 5 and 6 pieces tables) ##
##############################################################################

import os
import shutil
import chess.syzygy


def syzygy():
    for pop in range(3, 7):
        print(pop, "pieces")
        
        path = f'./set/popcount-{pop}.txt'
        save = f'./set/popcount-{pop}-original.txt'
        
        assert not os.path.exists(save)
        
        shutil.copy(path, save)
        
        if pop < 6:
            tablebase = "./syzygy/3-4-5"
        else:
            tablebase = "./syzygy/6-pieces"
        #end if
        
        with chess.syzygy.open_tablebase(tablebase) as syzygy:
            with open(save, 'rt') as i_file, open(path, 'wt') as o_file:
                for line in i_file:
                    pos, stm, pop, cnt, wr = line.strip().split()
                    
                    board = chess.Board(f'{pos} {stm} - - 0 1')
                    
                    wdl = syzygy.get_wdl(board)
                    
                    if wdl is None:
                        print(line.strip())
                        continue
                    #end if
                    
                    wr = {-2: "0.0", -1: "0.0", 0: "0.5", 1: "1.0", 2: "1.0"}[wdl]
                    
                    cnt = 1
                    
                    o_file.write(f'{pos} {stm} {pop} {cnt} {wr}\n')
                #end for
            #end with
        #end with
        
        print()
    #end for
#end def


if __name__ == "__main__":
    syzygy()
#end if
