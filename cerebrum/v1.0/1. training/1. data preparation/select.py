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
## NAME: select.py                                                          ##
## AUTHOR: David Carteau, France, March 2024                                ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Select training positions, store results in ./positions-shuffled.txt     ##
##############################################################################

import gc
import os
import random
import shutil

from tqdm import tqdm

# random seed = date of Orion's first public release :-)
random.seed(21052014)

# number of positions to select by popcount
# here 8M for 32-pieces + 8M for 31 pieces + ... + 8M for 3 pieces
N = 8*1024*1024


def select():
    positions = []
    
    for pop in range(32, 2, -1):
        print(pop, "pieces")
        
        mode = 'wt' if (pop == 32) else 'at'
        
        selection = []
        w_lines, d_lines, l_lines = [], [], []
        
        with open(f'./set/popcount-{pop}.txt', 'rt') as i_file:
            for line in tqdm(i_file):
                fen, stm, pop, cnt, wr = line.split()
                
                if cnt == "1":
                    if wr == "1.0":
                        w_lines.append(line)
                    elif wr == "0.5":
                        d_lines.append(line)
                    elif wr == "0.0":
                        l_lines.append(line)
                    else:
                        assert False
                    #end if
                else:
                    selection.append(line)
                #end if
            #end for
        #end with
        
        random.shuffle(selection)
        random.shuffle(w_lines)
        random.shuffle(d_lines)
        random.shuffle(l_lines)
        
        n = N - len(selection)
        
        if n > 0:
            print("MULTI:", len(selection))
            print("UNIQUE:")
            print("- W:", len(w_lines[:n // 3 + 1]))
            print("- D:", len(d_lines[:n // 3 + 1]))
            print("- L:", len(l_lines[:n // 3 + 1]))
            
            selection += w_lines[:n // 3 + 1]
            selection += d_lines[:n // 3 + 1]
            selection += l_lines[:n // 3 + 1]
        #end if
        
        selection = selection[:N]
        
        random.shuffle(selection)
        
        print("FINAL:", len(selection))
        print()
        
        with open(f'./positions.txt', mode) as o_file:
            o_file.write("".join(selection))
        #end with
        
        selection, w_lines, d_lines, l_lines = None, None, None, None
        gc.collect()
    #end for
    
    with open(f'./positions.txt', 'rt') as i_file:
        positions = i_file.readlines()
    #end with
    
    random.shuffle(positions)
    
    with open(f'./positions-shuffled.txt', 'wt') as o_file:
        for line in positions:
            o_file.write(line)
        #end for
    #end with
#end def


if __name__ == "__main__":
    select()
#end if
