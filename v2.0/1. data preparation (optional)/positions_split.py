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
## NAME: positions_split.py                                                 ##
## AUTHOR: David Carteau, France, November 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Split provided fenstrings by popcount, store result in ./txt folder      ##
##############################################################################

import os
import re
import gzip
import shutil


def popcount(position):
    return len(re.sub('[^pnbrqkPNBRQK]', '', position))
#end def


def save(lines, files):
    positions = {}
    
    for line in lines:
        line = line.split()
        
        fen, stm, cas, enp = line[:4]
        
        popcnt = popcount(fen)
        
        if popcnt == 2:
            continue
        #end if
        
        result = line[-1].rstrip(';')
        position = f'{fen} {stm} {cas} {enp},{result}\n'
        
        positions[position] = popcnt
    #end for
    
    for position, popcnt in positions.items():
        files[popcnt].write(position)
    #end for
#end def


def main():
    print('STEP 2: SPLIT POSITIONS...')
    print()
    
    path = './txt'
    
    if os.path.exists(path):
        shutil.rmtree(path)
    #end if
    
    os.mkdir(path)
    
    lines = []
    files = {i: gzip.open(f'{path}/popcount-{i}.txt.gz', 'wt', compresslevel=6) for i in range(3, 33)}
    
    while True:
        try:
            line = input().strip()
        except:
            break
        #end try
        
        if len(line) == 0:
            continue
        #end if
        
        if line.startswith('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq'):
            save(lines, files)
            lines = []
        #end if
        
        lines.append(line)
    #end while
    
    save(lines, files)
    lines = []
    
    for file in files:
        files[file].close()
    #end for
    
    print()
    print()
#end def


if __name__ == '__main__':
    main()
#end if
