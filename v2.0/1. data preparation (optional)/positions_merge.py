"""
The Cerebrum library
Copyright (c) 2020-2025, by David Carteau. All rights reserved.

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
## NAME: positions_merge.py                                                 ##
## AUTHOR: David Carteau, France, November 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Merge similar fenstrings, compute score, store result in ./set folder    ##
##############################################################################

import os
import gzip
import math
import shutil

from tqdm import tqdm


# https://www.chessprogramming.org/Pawn_Advantage,_Win_Percentage,_and_Elo

PIECES = 'PNBRQKpnbrqk'
VALUES = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0, -1.0, -3.0, -3.0, -5.0, -9.0, 0.0]

MAPPING_RESULTS = {'1-0': 'w', '0-1': 'b', '1/2-1/2': 'd'}


def main():
    print('STEP 3: MERGE POSITIONS...')
    print()
    
    path = './set'
    
    if os.path.exists(path):
        shutil.rmtree(path)
    #end if
    
    os.mkdir(path)
    
    for popcnt in range(3, 33):
        print(popcnt, 'pieces')
        
        results = {}
        
        with gzip.open(f'./txt/popcount-{popcnt}.txt.gz', 'rt') as i_file:
            for line in tqdm(i_file, unit_scale=True):
                position, result = line.strip().split(',')
                
                if result not in MAPPING_RESULTS:
                    continue
                #end if
                
                result = MAPPING_RESULTS[result]
                
                if position not in results:
                    results[position] = {'w': 0, 'b': 0, 'd': 0}
                #end if
                
                results[position][result] += 1
            #end for
        #end with
        
        print(f'{len(results):,} unique positions')
        
        with gzip.open(f'./set/popcount-{popcnt}.txt.gz', 'wt', compresslevel=6) as o_file:
            for position in results:
                cnt = sum(results[position].values())
                
                fen, stm, cas, enp = position.split()
                
                w = results[position][stm] / cnt
                d = results[position]['d'] / cnt
                
                # win ratio
                wr = w + (0.5 * d)
                
                # convert win ratio to [-9.0, 9.0] values
                wr = max(0.001, wr)
                wr = min(0.999, wr)
                wr = 3.0 * math.log10(wr / (1.0 - wr))
                
                # material
                mt = [VALUES[PIECES.find(piece)] for piece in fen if piece in PIECES]
                mt = sum(mt)
                
                if stm == 'b':
                    mt *= -1.0
                #end if
                
                # combination of win ratio and material
                evl = (0.5 * wr) + (0.5 * mt)
                
                position = f'{fen} {stm} {cas} {enp},{popcnt},{cnt},{evl:.4f}\n'
                
                o_file.write(position)
            #end for
        #end with
        
        print()
#end def


if __name__ == '__main__':
    main()
#end if
