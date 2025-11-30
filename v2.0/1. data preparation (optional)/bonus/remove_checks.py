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
## NAME: remove_checks.py                                                   ##
## AUTHOR: David Carteau, France, November 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Remove positions positions in check (optional)                           ##
##############################################################################

import tqdm
import chess


def main():
    with open('./positions-shuffled.txt', 'rt') as i_file:
        with open('./positions-shuffled-without-checks.txt', 'wt') as o_file:
            for line in tqdm.tqdm(i_file, unit_scale=True):
                fen, stm, pop, cnt, evl = line.split()
                
                board = chess.Board(f'{fen} {stm} - -')
                
                if not board.is_check():
                    o_file.write(line)
                #end if
            #end for
        #end with
    #end with
#end def


if __name__ == "__main__":
    main()
#end if
