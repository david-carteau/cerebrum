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
## NAME: shuffle.py                                                         ##
## AUTHOR: David Carteau, France, February 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Shuffle positions.txt file (result stored in positions-shuffled.txt)     ##
##############################################################################

import os
import random

from tqdm import tqdm

# random seed (adjust if needed)
# 21.05.2014 = date of Orion's first public release :-)
SEED = 21052014

def main():
    source = './positions.txt'
    target = './positions-shuffled.txt'
    
    pointers = []
    
    print("Reading dataset...")
    
    with open(source, 'rt') as file:
        pointer = 0
        
        for line in tqdm(file):
            pointers.append(pointer)
            pointer += len(line) + 1
        #end for
    #end with
    
    print()
    
    # shuffle the pointers (this is more memory efficient than loading plain text positions into a list
    # and shuffling the list, since we are only shuffling a list of numbers)
    
    print("Shuffling samples...")
    
    random.seed(SEED)
    random.shuffle(pointers)
    
    print("Done !")
    print()
    
    print("Saving shuffled dataset...")
    
    with open(source, 'rt') as i_file, open(target, 'wt') as o_file:
        for pointer in tqdm(pointers):
            i_file.seek(pointer)
            line = i_file.readline()
            o_file.write(line)
        #end for
    #end with
    
    print("Done !")
    print()
#end def

if __name__ == "__main__":
    main()
#end if
