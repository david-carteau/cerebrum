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
## NAME: merge.py                                                           ##
## AUTHOR: David Carteau, France, February 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Merge similar fenstrings, adjust win ratio, store result in ./set folder ##
##############################################################################


import os
import shutil

from tqdm import tqdm


def merge():
    path = "./set"
    
    if os.path.exists(path):
        shutil.rmtree(path)
    #end if
    
    os.mkdir(path)
    
    for pop in range(32, 2, -1):
        print(pop, "pieces")
        
        results = {}
        
        with open(f'./txt/popcount-{pop}.txt', 'rt') as i_file:
            for line in tqdm(i_file):
                fen, stm, res = line.split()
                
                line = fen + " " + stm
                
                try:
                    result = {
                        "1-0": "w",
                        "0-1": "b",
                        "1/2-1/2": "d"
                    }[res]
                except:
                    continue
                #end try
                
                if line not in results:
                    results[line] = {"w": 0, "b": 0, "d": 0}
                #end if
                
                results[line][result] += 1
            #end for
        #end with
        
        print(len(results), "unique positions")
        
        with open(f'./set/popcount-{pop}.txt', 'wt') as o_file:
            for line in results:
                cnt = sum(results[line].values())
                
                fen, stm = line.split()
                
                w = results[line][stm] / cnt
                d = results[line]["d"] / cnt
                
                wr = w + 0.5 * d
                
                line = " ".join([fen, stm, str(pop), str(cnt), str(wr)]) + "\n"
                
                o_file.write(line)
            #end for
        #end with
        
        print()
#end def


if __name__ == "__main__":
    merge()
#end if
