/*
 * The Cerebrum library and engine
 * Copyright (c) 2025, by David Carteau. All rights reserved.
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
/** NAME: cerebrum.h (inference)                                           **/
/** AUTHOR: David CARTEAU, France, March 2025                              **/
/** LICENSE: MIT (see above and "license.txt" file content)                **/
/****************************************************************************/

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

// uncomment the following line if needed (debug mode)
//#define NN_DEBUG

// name of the default neural network file
#define NN_FILE "cerebrum-v1.1.nn"

#define None -1

// network architecture
#define NN_SIZE_L0 768
#define NN_SIZE_L1 128
#define NN_SIZE_L2 32
#define NN_SIZE_L3 2
#define NN_SIZE_L4 None


/****************************************************************************/
/** TYPE DEFINITIONS                                                       **/
/****************************************************************************/

typedef int16_t NN_Accumulator[2][NN_SIZE_L1];


/****************************************************************************/
/** PUBLIC FUNCTIONS                                                       **/
/****************************************************************************/

int nn_convert(void);
int nn_load(char* filename);

void nn_init_accumulator(NN_Accumulator accumulator);

void nn_add_piece(NN_Accumulator accumulator, int piece_type, int piece_color, int piece_position);
void nn_del_piece(NN_Accumulator accumulator, int piece_type, int piece_color, int piece_position);
void nn_mov_piece(NN_Accumulator accumulator, int piece_type, int piece_color, int from, int to);

void nn_update_all_pieces(NN_Accumulator accumulator, const uint64_t* whites, const uint64_t* blacks);

float nn_evaluate(NN_Accumulator accumulator, int color);

#endif // CEREBRUM_H_INCLUDED
