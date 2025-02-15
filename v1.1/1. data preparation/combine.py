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
## NAME: combine.py                                                         ##
## AUTHOR: David Carteau, France, February 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Combine pgn files located in ./pgn folder, store result in ./games.pgn   ##
##############################################################################


import os


def combine():
    path = "./pgn"
    
    files = []
    
    for file in os.scandir(path):
        if file.name.lower().endswith(".pgn"):
            files.append(f'{path}/{file.name}')
        #end if
    #end for
    
    files = sorted(files)
    
    with open("./games.pgn", "wt") as o_file:
        for file in files:
            print(file)
            
            with open(file, "rt") as i_file:
                for line in i_file:
                    o_file.write(line)
                #end for
            #end with
        #end for
    #end with
#end def


if __name__ == "__main__":
    combine()
#end if
