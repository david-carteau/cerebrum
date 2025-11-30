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
## NAME: positions_shuffle.py                                               ##
## AUTHOR: David Carteau, France, November 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Shuffle positions.txt file (result in positions-shuffled.txt)            ##
##############################################################################

import os
import shutil
import random

from tqdm import tqdm


# number of temporary files (adjust if needed)
N_CHUNCKS = 250

# name of the temporary folder (adjust if needed)
TMP_FOLDER = os.path.abspath('./tmp')

# random seed (adjust if needed)
# 21.05.2014 = date of Orion's first public release :-)
SEED = 21052014


def main():
    print('STEP 5: SHUFFLE POSITIONS...')
    print()
    
    random.seed(SEED)
    
    source = './positions.txt'
    target = './positions-shuffled.txt'
    
    # temporary folder creation
    
    if os.path.exists(TMP_FOLDER):
        shutil.rmtree(TMP_FOLDER)
    #end if
    
    os.mkdir(TMP_FOLDER)
    
    # 1st step : read source file and spread its content over temporary files
    
    print('Reading dataset...')
    
    chuncks = []
    
    for i in range(N_CHUNCKS):
        chunck = open(f'{TMP_FOLDER}/chunck-{i}.txt', 'wt')
        chuncks.append(chunck)
    #end for
    
    with open(source, 'rt') as file:
        for line in tqdm(file, unit_scale=True):
            i = random.randrange(N_CHUNCKS)
            chuncks[i].write(line)
        #end for
    #end with
    
    for i in range(N_CHUNCKS):
        chuncks[i].close()
    #end for
    
    print()
    
    # 2nd step : read each chunk, shuffle its content, and write to target file
    
    print('Shuffling samples...')
    
    with open(target, 'wt') as o_file:
        for i in tqdm(range(N_CHUNCKS)):
            with open(f'{TMP_FOLDER}/chunck-{i}.txt', 'rt') as i_file:
                lines = i_file.readlines()
            #end with
            
            random.shuffle(lines)
            
            for line in lines:
                o_file.write(line)
            #end for
        #end for
    #end with
    
    print()
    
    # temporary folder deletion
    
    shutil.rmtree(TMP_FOLDER)
    
    print('Done!')
#end def


if __name__ == '__main__':
    main()
#end if
