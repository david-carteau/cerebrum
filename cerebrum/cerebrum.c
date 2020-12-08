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
/** NAME : cerebrum.c                                                      **/
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
 
/*
 * Example of usage (in Orion UCI engine) :
 *
 * 1) when starting :
 * NN_Network nn;
 * NN_Board board;
 *
 * 2) when initialising a position :
 * board.pieces[0] = &white.pieces[PAWN];
 * board.pieces[1] = &black.pieces[PAWN];
 *
 * 3) when making move :
 * memcpy(&save, &board, sizeof(board));
 *
 * if (moved_piece_type == KING || promotion || capture_en_passant) {
 *     board.pieces[0] = &white.pieces[PAWN];
 *     board.pieces[1] = &black.pieces[PAWN];
 *     nn_inputs_upd_all(&nn, &board);
 * } else {
 *     nn_inputs_mov_piece(&nn, &board, moved_piece_type, (color == WHITE ? 0 : 1), from, to);
 *
 *     if (capture) {
 *         nn_inputs_del_piece(&nn, &board, captured_piece_type, (color == WHITE ? 1 : 0), to);
 *     }
 * }
 *
 * 4) when unmaking move :
 * memcpy(&board, &save, sizeof(save));
 *
 * 5) when evaluating :
 * nn_eval(&nn, &board, (color == WHITE ? 0 : 1));
 */
 
#include "cerebrum.h"

static NN_Storage storage;
static NN_Storage* st = &storage;



/****************************************************************************/
/* I/O FUNCTIONS                                                            */
/****************************************************************************/

static int nn_convert(char* filename) {
    FILE* file = fopen("network.txt", "r");
    
    if (file == NULL) {
        return -1;
    }
    
    /* load network (text format) */
    
    float value;
    char line[256];
    
    // W0
    for (int col = 0; col < NN_SIZE; col++) {
        for (int row = 0; row < 40960; row++) {
            if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1 || value < -THRESHOLD || value > THRESHOLD) {
                fclose(file);
                return -1;
            }
            st->W0[row * NN_SIZE + col] = (int16_t) (value * FACTOR);
        }
    }
    
    // B0
    for (int row = 0; row < NN_SIZE; row++) {
        if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
            fclose(file);
            return -1;
        }
        st->B0[row] = value;
    }
    
    // W1
    for (int col = 0; col < 32; col++) {
        for (int row = 0; row < (NN_SIZE*2); row++) {
            if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
                fclose(file);
                return -1;
            }
            st->W1[row * 32 + col] = value;
        }
    }
    
    // B1
    for (int row = 0; row < 32; row++) {
        if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
            fclose(file);
            return -1;
        }
        st->B1[row] = value;
    }
    
    // W2
    for (int col = 0; col < 32; col++) {
        for (int row = 0; row < 32; row++) {
            if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
                fclose(file);
                return -1;
            }
            st->W2[row * 32 + col] = value;
        }
    }
    
    // B2
    for (int row = 0; row < 32; row++) {
        if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
            fclose(file);
            return -1;
        }
        st->B2[row] = value;
    }

    // W3
    for (int col = 0; col < 1; col++) {
        for (int row = 0; row < 32; row++) {
            if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
                fclose(file);
                return -1;
            }
            st->W3[row + col] = value;
        }
    }
    
    // B3
    for (int row = 0; row < 1; row++) {
        if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
            fclose(file);
            return -1;
        }
        st->B3[row] = value;
    }
    
    fclose(file);
    
    /* W1 & W2 transposition */
    /* (optional, but allows intrinsics usage by aligning data) */
    
    float W1[NN_SIZE*2*32];
    float W2[32*32];
    
    memcpy(W1, st->W1, sizeof(st->W1));
    memcpy(W2, st->W2, sizeof(st->W2));
    
    for (int col = 0; col < 32; col++) {
        for (int row = 0; row < (NN_SIZE*2); row++) {
            st->W1[col * (NN_SIZE*2) + row] = W1[row * 32 + col];
        }
    }
    
    for (int col = 0; col < 32; col++) {
        for (int row = 0; row < 32; row++) {
            st->W2[col * 32 + row] = W2[row * 32 + col];
        }
    }
    
    /* save network (binary format) */
    
    file = fopen(filename, "wb");
    
    if (file == NULL) {
        return -1;
    }
    
    fwrite(st, sizeof(NN_Storage), 1, file);
    
    fclose(file);
    
    return 0;
}

int nn_load(NN_Network* nn, char* filename) {
    *nn = (NN_Network) {0};
    *st = (NN_Storage) {0};
    
    FILE* file = fopen(filename, "rb");
    
    if (file == NULL) {
        printf("info debug NN file conversion...\n");
        if (nn_convert(filename) == -1) {
            return -1;
        }
    }
    
    file = fopen(filename, "rb");
    
    if (file == NULL) {
        return -1;
    }
    
    fread(st, sizeof(NN_Storage), 1, file);
    
    for (size_t i = 0; i < (sizeof(st->W0) / sizeof(st->W0[0])); i++) {
        nn->W0[i] = st->W0[i] / FACTOR;
    }
    
    size_t size = sizeof(st->B0);
    size += sizeof(st->W1) + sizeof(st->B1);
    size += sizeof(st->W2) + sizeof(st->B2);
    size += sizeof(st->W3) + sizeof(st->B3);
    
    memcpy(nn->B0, st->B0, size);
    
    fclose(file);
    
    return 0;
}



/****************************************************************************/
/* HELPERS                                                                  */
/****************************************************************************/

#define NN_GET_POSITION(pieces) __builtin_ctzll(pieces)
#define NN_POP_POSITION(pieces) pieces &= pieces - 1

static float clamp(float value) {
    if (value < NN_RELU_MIN) {
        return NN_RELU_MIN;
    }
    if (value > NN_RELU_MAX) {
        return NN_RELU_MAX;
    }
    return value;
}



/****************************************************************************/
/* EVALUATION                                                               */
/****************************************************************************/

/* B = Biases, I = Input layer, W = Weights, O = Output layer */
/* idim/odim = size of input/output layers (i.e. the number of neurons) */

static void nn_compute_layer(float* B, float* I, float* W, float* O, int idim, int odim, int with_relu) {
    #if defined(NN_DEBUG)
        assert(idim > 0 && odim > 0 && (with_relu == 0 || with_relu == 1));
    #end if
    
    for (int o = 0; o < odim; o++) {
        float sum = B[o];
        
        // intrinsics dot product (simple, but poor performance)
        // see: https://www.codesd.com/item/how-to-access-the-components-of-the-vector-ps-of-256-bits.html
        
        /*
        const int offset = o * idim;
        
        for (int i = 0; i < idim; i += 8) {
            float v[8]; __m256* ptr_v = (__m256*) v;
            *ptr_v = _mm256_dp_ps(_mm256_load_ps(&I[i]), _mm256_load_ps(&W[offset + i]), 0xff); 
            sum += v[0] + v[4];
        }
        */
        
        #if defined(NN_WITH_FMA)
            // intrinsics dot product (more complex, but performance gain)
            // see: https://stackoverflow.com/questions/59494745/avx2-computing-dot-product-of-512-float-arrays
            
            #if defined(NN_DEBUG)
                // input layer must contain a multiple of 32 neurons to allow parallel dot product
                assert( (idim % 32) == 0 );
            #end if
            
            const int offset = o * idim;
            
            __m256 dot0 = _mm256_mul_ps( _mm256_loadu_ps(&I[ 0]), _mm256_loadu_ps(&W[offset +  0]) );
            __m256 dot1 = _mm256_mul_ps( _mm256_loadu_ps(&I[ 8]), _mm256_loadu_ps(&W[offset +  8]) );
            __m256 dot2 = _mm256_mul_ps( _mm256_loadu_ps(&I[16]), _mm256_loadu_ps(&W[offset + 16]) );
            __m256 dot3 = _mm256_mul_ps( _mm256_loadu_ps(&I[24]), _mm256_loadu_ps(&W[offset + 24]) );
            
            for (int i = 32; i < idim; i += 32) {
                dot0 = _mm256_fmadd_ps( _mm256_loadu_ps(&I[i +  0]), _mm256_loadu_ps(&W[offset + i +  0]), dot0 );
                dot1 = _mm256_fmadd_ps( _mm256_loadu_ps(&I[i +  8]), _mm256_loadu_ps(&W[offset + i +  8]), dot1 );
                dot2 = _mm256_fmadd_ps( _mm256_loadu_ps(&I[i + 16]), _mm256_loadu_ps(&W[offset + i + 16]), dot2 );
                dot3 = _mm256_fmadd_ps( _mm256_loadu_ps(&I[i + 24]), _mm256_loadu_ps(&W[offset + i + 24]), dot3 );
            }
            
            const __m256 dot01 = _mm256_add_ps( dot0, dot1 );
            const __m256 dot23 = _mm256_add_ps( dot2, dot3 );
            const __m256 dot03 = _mm256_add_ps( dot01, dot23 );
            const __m128 r4 = _mm_add_ps( _mm256_castps256_ps128( dot03 ), _mm256_extractf128_ps( dot03, 1 ) );
            const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps( r4, r4 ) );
            const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );
            
            sum += _mm_cvtss_f32( r1 );
        #else
            // naive dot product
            
            for (int i = 0; i < idim; i++) {
                sum += W[o * idim + i] * I[i]; // without transposition : sum += W[i * odim + o] * I[i];
            }
        #endif
        
        // ReLU ?
        
        if (with_relu) {
            O[o] = clamp(sum);
        } else {
            O[o] = sum;
        }
    }
}

int nn_eval(NN_Network* nn, NN_Board* board, int color) {
    #if defined(NN_DEBUG)
        assert(color == 0 || color == 1);
    #endif
    
    // layer 0 (features transformation with ReLU)
    
    float O0[NN_SIZE*2];
    
    for (int o = 0; o < NN_SIZE; o++) {
        O0[o          ] = clamp(board->accumulator[color][o]);
        O0[o + NN_SIZE] = clamp(board->accumulator[1 - color][o]);
    }
    
    // layers 1 & 2 (with ReLU), layer 3 (without ReLU)
    
    float O1[32], O2[32], O3[1];
    
    nn_compute_layer(nn->B1, O0, nn->W1, O1, NN_SIZE*2, 32, 1);
    nn_compute_layer(nn->B2, O1, nn->W2, O2, 32, 32, 1);
    nn_compute_layer(nn->B3, O2, nn->W3, O3, 32, 1, 0);
    
    // final evaluation (in centipawns) !
    
    return (int) (O3[0] * 100);
}

/* *************************************************************** */

void nn_inputs_upd_all(NN_Network* nn, NN_Board* board) {
    /*
    for (int o = 0; o < NN_SIZE; o++) {
        board->accumulator[0][o] = nn->B0[o];
        board->accumulator[1][o] = nn->B0[o];
    }
    */
    
    memcpy(board->accumulator[0], nn->B0, sizeof(nn->B0));
    memcpy(board->accumulator[1], nn->B0, sizeof(nn->B0));
    
    for (int piece_color = 0; piece_color <= 1; piece_color++) {
        for (int piece_type = 0; piece_type <= 4; piece_type++) {
            uint64_t pieces = board->pieces[piece_color][piece_type];
            
            while (pieces) {
                const int piece_position = NN_GET_POSITION(pieces);
                nn_inputs_add_piece(nn, board, piece_type, piece_color, piece_position);
                NN_POP_POSITION(pieces);
            }
        }
    }
}

void nn_inputs_add_piece(NN_Network* nn, NN_Board* board, int piece_type, int piece_color, int piece_position) {
    const int white_king_position = NN_GET_POSITION(board->pieces[0][5]);
    const int black_king_position = NN_GET_POSITION(board->pieces[1][5]) ^ 63;
    
    #if defined(NN_DEBUG)
        assert(piece_type >= 0 && piece_type <= 4);
        assert(piece_color >= 0 && piece_color <= 1);
        assert(piece_position >= 0 && piece_position <= 63);
        assert(white_king_position >= 0 && white_king_position <= 63);
        assert(black_king_position >= 0 && black_king_position <= 63);
    #endif
    
    const int index_w = (piece_type << 1) + (piece_color);
    const int index_b = (piece_type << 1) + (1 - piece_color);
    
    #if defined(NN_DEBUG)
        assert(index_w >= 0 && index_w <= 9);
        assert(index_b >= 0 && index_b <= 9);
    #endif
    
    const int sq_w = piece_position;
    const int sq_b = piece_position ^ 63;
    
    const int feature_w = (640 * white_king_position) + (64 * index_w) + (sq_w);
    const int feature_b = (640 * black_king_position) + (64 * index_b) + (sq_b);
    
    #if defined(NN_DEBUG)
        assert(feature_w >= 0 && feature_w <= 40960);
        assert(feature_b >= 0 && feature_b <= 40960);
    #endif
    
    for (int o = 0; o < NN_SIZE; o++) {
        board->accumulator[0][o] += nn->W0[NN_SIZE * feature_w + o];
        board->accumulator[1][o] += nn->W0[NN_SIZE * feature_b + o];
    }
}

void nn_inputs_del_piece(NN_Network* nn, NN_Board* board, int piece_type, int piece_color, int piece_position) {
    const int white_king_position = NN_GET_POSITION(board->pieces[0][5]);
    const int black_king_position = NN_GET_POSITION(board->pieces[1][5]) ^ 63;
    
    #if defined(NN_DEBUG)
        assert(piece_type >= 0 && piece_type <= 4);
        assert(piece_color >= 0 && piece_color <= 1);
        assert(piece_position >= 0 && piece_position <= 63);
        assert(white_king_position >= 0 && white_king_position <= 63);
        assert(black_king_position >= 0 && black_king_position <= 63);
    #endif
    
    const int index_w = (piece_type << 1) + (piece_color);
    const int index_b = (piece_type << 1) + (1 - piece_color);
    
    #if defined(NN_DEBUG)
        assert(index_w >= 0 && index_w <= 9);
        assert(index_b >= 0 && index_b <= 9);
    #endif
    
    const int sq_w = piece_position;
    const int sq_b = piece_position ^ 63;
    
    const int feature_w = (640 * white_king_position) + (64 * index_w) + (sq_w);
    const int feature_b = (640 * black_king_position) + (64 * index_b) + (sq_b);
    
    #if defined(NN_DEBUG)
        assert(feature_w >= 0 && feature_w <= 40960);
        assert(feature_b >= 0 && feature_b <= 40960);
    #endif
    
    for (int o = 0; o < NN_SIZE; o++) {
        board->accumulator[0][o] -= nn->W0[NN_SIZE * feature_w + o];
        board->accumulator[1][o] -= nn->W0[NN_SIZE * feature_b + o];
    }
}

void nn_inputs_mov_piece(NN_Network* nn, NN_Board* board, int piece_type, int piece_color, int from, int to) {
    const int white_king_position = NN_GET_POSITION(board->pieces[0][5]);
    const int black_king_position = NN_GET_POSITION(board->pieces[1][5]) ^ 63;
    
    #if defined(NN_DEBUG)
        assert(piece_type >= 0 && piece_type <= 4);
        assert(piece_color >= 0 && piece_color <= 1);
        assert(from >= 0 && from <= 63 && to >= 0 && to <= 63);
        assert(white_king_position >= 0 && white_king_position <= 63);
        assert(black_king_position >= 0 && black_king_position <= 63);
    #endif
    
    const int index_w = (piece_type << 1) + (piece_color);
    const int index_b = (piece_type << 1) + (1 - piece_color);
    
    #if defined(NN_DEBUG)
        assert(index_w >= 0 && index_w <= 9);
        assert(index_b >= 0 && index_b <= 9);
    #endif
    
    const int fr_w = from;
    const int fr_b = from ^ 63;
    
    const int to_w = to;
    const int to_b = to ^ 63;
    
    const int feature_w_fr = (640 * white_king_position) + (64 * index_w) + (fr_w);
    const int feature_b_fr = (640 * black_king_position) + (64 * index_b) + (fr_b);
    
    const int feature_w_to = (640 * white_king_position) + (64 * index_w) + (to_w);
    const int feature_b_to = (640 * black_king_position) + (64 * index_b) + (to_b);
    
    #if defined(NN_DEBUG)
        assert(feature_w_fr >= 0 && feature_w_fr <= 40960);
        assert(feature_b_fr >= 0 && feature_b_fr <= 40960);
        assert(feature_w_to >= 0 && feature_w_to <= 40960);
        assert(feature_b_to >= 0 && feature_b_to <= 40960);
    #endif
    
    for (int o = 0; o < NN_SIZE; o++) {
        board->accumulator[0][o] += nn->W0[NN_SIZE * feature_w_to + o];
        board->accumulator[0][o] -= nn->W0[NN_SIZE * feature_w_fr + o];
        
        board->accumulator[1][o] += nn->W0[NN_SIZE * feature_b_to + o];
        board->accumulator[1][o] -= nn->W0[NN_SIZE * feature_b_fr + o];
    }
}
