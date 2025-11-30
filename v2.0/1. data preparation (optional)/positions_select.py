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
## NAME: positions_select.py                                                ##
## AUTHOR: David Carteau, France, November 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Select training positions, store results in ./positions.txt              ##
##############################################################################

import gzip
from tqdm import tqdm


# N = number of positions to select (adjust if needed)
# i.e. N for 3-pieces + N for 4-pieces + ... + N for 32-pieces
N = 8 * 1024 * 1024


def select(n, samples, selection):
    n = 1 + (n // len(samples))
    
    for evl in samples:
        selection += samples[evl][:n]
        
        if len(samples[evl]) <= n:
            samples[evl] = []
        else:
            samples[evl] = samples[evl][n:]
        #end if
    #end for
    
    samples = {k: v for k, v in samples.items() if len(v)}
    
    return samples, selection
#end def


def main():
    print('STEP 4: SELECT POSITIONS...')
    print()
    
    positions = []
    
    for popcnt in range(3, 33):
        print(popcnt, 'pieces')
        
        selection = []
        samples = {}
        
        with gzip.open(f'./set/popcount-{popcnt}.txt.gz', 'rt') as i_file:
            for line in tqdm(i_file, unit_scale=True):
                position, popcnt, cnt, evl = line.strip().split(',')
                
                sample = f'{position},{evl}\n'
                
                if cnt == '1':
                    evl = float(evl)
                    
                    if evl not in samples:
                        samples[evl] = set()
                    #end if
                    
                    samples[evl].add(sample)
                else:
                    selection.append(sample)
                #end if
            #end for
        #end with
        
        samples = {k: list(v) for k, v in samples.items()}
        
        while True:
            s = len(selection)
            r = sum([len(v) for k, v in samples.items()])
            n = N - len(selection)
            v = len(samples)
            
            print(f'Selected: {s:,}', end=' - ')
            print(f'Remaining: {r:,}', end=' - ')
            print(f'To select: {n:,}', end=' - ')
            print(f'Values: {v:,}')
            
            if s < N and r > 0:
                samples, selection = select(n, samples, selection)
            else:
                break
            #end if
        #end while
        
        selection = selection[:N]
        
        print(f'FINAL: {len(selection):,}')
        print()
        
        mode = 'wt' if (popcnt == 3) else 'at'
        
        with open(f'./positions.txt', mode) as o_file:
            for sample in selection:
                o_file.write(sample)
            #end for
        #end with
    #end for
#end def


if __name__ == '__main__':
    main()
#end if
