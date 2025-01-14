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
## NAME: select.py                                                          ##
## AUTHOR: David Carteau, France, January 2025                              ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Select training positions, store results in ./positions-shuffled.txt     ##
##############################################################################


import gc
import random

from tqdm import tqdm


# random seed = date of Orion's first public release :-)
random.seed(21052014)

# N = number of positions to select by popcount
# i.e. N for 3-pieces + N for 4-pieces + ... + N for 32-pieces
N = 8*1024*1024


def select():
    positions = []
    
    for pop in range(3, 33):
        print(pop, "pieces")
        
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
        
        mode = 'wt' if (pop == 3) else 'at'
        
        with open(f'./positions.txt', mode) as o_file:
            o_file.write("".join(selection))
        #end with
        
        selection, w_lines, d_lines, l_lines = None, None, None, None
        gc.collect()
    #end for
#end def


if __name__ == "__main__":
    select()
#end if
