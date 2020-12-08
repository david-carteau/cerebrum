/*
 * The Cerebrum library
 * Copyright (c) 2020, by David Carteau. All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/****************************************************************************/
/** NAME : cerebrum.h                                                      **/
/** AUTHOR : David Carteau, France, December 2020                          **/
/** LICENSE: MIT (see above and "license.txt" file content)                **/
/****************************************************************************/

/* 
 * The library is composed of two parts :
 * - cerebrum.py : Python program which aims at training neural networks
 *   used in the Orion UCI chess engine
 * - cerebrum.h and cerebrum.c : inference code which can be embedded in a
 *   chess engine
 */

#ifndef CEREBRUM_H_INCLUDED
#define CEREBRUM_H_INCLUDED

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <inttypes.h>
#include "immintrin.h"



/****************************************************************************/
/** MACROS                                                                 **/
/****************************************************************************/

// The two following can also be added as compilation flags
// e.g. add : -D NN_DEBUG and/or -D NN_WITH_FMA to gcc / clang

// allows assertions helping to debug when embedding the library
//#define NN_DEBUG

// allows the use of intrinsics, to increase speed of dot products
//#define NN_WITH_FMA

// name of the default network file
#define NN_FILE "network.nn"

// output size of the first layer, in neurons
// set it to 256 to have an architecture similar to Stockfish's one
#define NN_SIZE 128

// constants used to convert first layer weights to int16
#define THRESHOLD 3.2f
#define FACTOR 10000.0f

// boundaries for layers outputs
#define NN_RELU_MIN 0.0f
#define NN_RELU_MAX 1.0f



/****************************************************************************/
/** TYPE DEFINITIONS                                                       **/
/****************************************************************************/

typedef struct {
    float W0[40960*NN_SIZE];
    float B0[NN_SIZE];
    float W1[NN_SIZE*2*32];
    float B1[32];
    float W2[32*32];
    float B2[32];
    float W3[32*1];
    float B3[1];
} NN_Network;

typedef struct {
    int16_t W0[40960*NN_SIZE];
    float B0[NN_SIZE];
    float W1[NN_SIZE*2*32];
    float B1[32];
    float W2[32*32];
    float B2[32];
    float W3[32*1];
    float B3[1];
} NN_Storage;

/*
 * pieces must be stored by color and by type, under the bitboard format, with
 * the following convention:
 * - white = 0, black = 1
 * - pawn = index 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5
 */

typedef struct {
    uint64_t* pieces[2];
    float accumulator[2][NN_SIZE];
} NN_Board;



/****************************************************************************/
/** EXTERNAL FUNCTIONS                                                     **/
/****************************************************************************/

int nn_load(NN_Network* nn, char* filename);

void nn_inputs_upd_all(NN_Network* nn, NN_Board* board);
void nn_inputs_add_piece(NN_Network* nn, NN_Board* board, int piece_type, int piece_color, int piece_position);
void nn_inputs_del_piece(NN_Network* nn, NN_Board* board, int piece_type, int piece_color, int piece_position);
void nn_inputs_mov_piece(NN_Network* nn, NN_Board* board, int piece_type, int piece_color, int from, int to);

int nn_eval(NN_Network* nn, NN_Board* board, int color);

#endif // CEREBRUM_H_INCLUDED
