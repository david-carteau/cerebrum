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
## NAME: games_select.py                                                    ##
## AUTHOR: David Carteau, France, November 2025                             ##
## LICENSE: MIT (see above and "license.txt" file content)                  ##
##############################################################################

##############################################################################
## PURPOSE:                                                                 ##
## Select games from multiple pgn files, store results in ./games.pgn       ##
##############################################################################

import os
import tqdm
import random


# MAX_PLIES = maximum number of plies for selected games (adjust if needed)
MAX_PLIES = 127

# random seed (adjust if needed)
# 21.05.2014 = date of Orion's first public release :-)
SEED = 21052014


# usage: for _ in tqdm.tqdm(generator(), unit_scale=True):
def generator():
    while True:
        yield
    #end while
#end def


def read_games():
    files = []
    
    for file in os.scandir('./pgn'):
        if file.name.lower().endswith('.pgn'):
            files.append(f'./pgn/{file.name}')
        #end if
    #end for
    
    for pgn_file in sorted(files):
        with open(pgn_file, 'rt') as file:
            lines = []
            
            for line in file:
                if line.startswith('[Event ') and len(lines):
                    yield lines
                    lines = []
                else:
                    lines.append(line)
                #end if
            #end for
        #end with
        
        if len(lines):
            yield lines
        #end if
    #end for
    
    return None
#end def

def main():
    print('STEP 1: SELECT GAMES...')
    print()
    
    random.seed(SEED)
    
    mates = 0
    errors = 0
    
    draws = []
    
    with open('games.pgn', 'wt') as dst_file:
        games = read_games()
        
        for game in tqdm.tqdm(games, unit_scale=True):
            result = None
            plycount = None
            
            for line in game:
                if line.startswith('[Result "'):
                    result = line.split('"')[1]
                #end if
                if line.startswith('[PlyCount "'):
                    plycount = line.split('"')[1]
                #end if
            #end for
            
            if result is None or plycount is None:
                #print(game)
                errors += 1
                continue
            #end if
            
            if result not in ['1-0', '0-1', '1/2-1/2']:
                #print(game)
                errors += 1
                continue
            #end if
            
            if int(plycount) > MAX_PLIES:
                continue
            #end if
            
            game = ''.join(game)
            
            if result == '1/2-1/2':
                draws.append(game)
            else:
                mates += 1
                dst_file.write(game)
                dst_file.write('\n')
            #end if
        #end for
        
        random.shuffle(draws)
        index = int(mates / 2)
        draws = draws[:index]
        
        print(mates, 'mate(s)', '+', len(draws), 'draw(s)', '+', errors, 'error(s)')
        
        for game in draws:
            dst_file.write(game)
            dst_file.write('\n')
        #end for
        
        print()
    #end with
#end def

if __name__ =='__main__':
    main()
#end if
