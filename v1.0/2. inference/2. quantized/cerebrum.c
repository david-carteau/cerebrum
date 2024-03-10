/*
 * The Cerebrum library and engine
 * Copyright (c) 2024, by David Carteau. All rights reserved.
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
/** NAME: cerebrum.c (inference)                                           **/
/** AUTHOR: David Carteau, France, March 2024                              **/
/** LICENSE: MIT (see above and "license.txt" file content)                **/
/****************************************************************************/

/*
 * Simplified example of usage (to adapt to your engine) :
 *
 * Assuming that pieces are stored in bitboards
 * with WHITE=0, BLACK=1, PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
 * 
 * const uint64_t* whites = &white.pieces[PAWN];
 * const uint64_t* blacks = &black.pieces[PAWN];
 * 
 * 1) when starting :
 * #include "cerebrum.h"
 * NN_Accumulator accumulator;
 * nn_load(NN_FILE);
 *
 * 2) when initialising a position :
 * nn_update_all_pieces(accumulator, whites, blacks);
 *
 * 3) when making move :
 * memcpy(&save, &accumulator, sizeof(accumulator));
 *
 * if (castling || promotion || capture_en_passant) {
 *     nn_update_all_pieces(next_ply->accumulator, whites, blacks);
 * } else {
 *     nn_mov_piece(next_ply->accumulator, moved - 1, color == WHITE ? 0 : 1, from, to);
 *     
 *     if (captured) {
 *         nn_del_piece(next_ply->accumulator, captured - 1, color == WHITE ? 1 : 0, to);
 *     }
 * }
 *
 * 4) when unmaking move :
 * memcpy(&accumulator, &save, sizeof(save));
 *
 * 5) when evaluating :
 * nn_eval(accumulator, (color == WHITE ? 0 : 1));
 */

#include "cerebrum.h"

#define NN_GET_POSITION(pieces) __builtin_ctzll(pieces)
#define NN_POP_POSITION(pieces) pieces &= pieces - 1


/****************************************************************************/
/** TYPES & VARIABLES                                                      **/
/****************************************************************************/

typedef struct {
	char name[256];
	char author[256];
	
	float wr;
	float mt;
	
	int16_t W0[768 * NN_SIZE_L1];
	int16_t B0[NN_SIZE_L1];
	
	int8_t W1[NN_SIZE_L1 * 2 * NN_SIZE_L2];
	int8_t B1[NN_SIZE_L2];
	
	int8_t W2[NN_SIZE_L2 * NN_SIZE_L3];
	int8_t B2[NN_SIZE_L3];
	
	int8_t W3[NN_SIZE_L3 * NN_SIZE_L4];
	int8_t B3[NN_SIZE_L4];
} NN_Network;

static const int8_t FACTOR = 64;
static const int8_t THRESHOLD = 127;

static NN_Network network;
static NN_Network* nn = &network;


/****************************************************************************/
/** PRIVATE FUNCTIONS                                                      **/
/****************************************************************************/

static int8_t nn_clamp(int32_t sum, int8_t min, int8_t max) {
	sum /= FACTOR;
	
	if (sum < min) {
		return min;
	}
	
	if (sum > max) {
		return max;
	}
	
	return (int8_t) (sum);
}

static int8_t nn_clamp_acc(int16_t sum) {
	if (sum < 0) {
		return 0;
	}
	
	if (sum > THRESHOLD) {
		return THRESHOLD;
	}
	
	return (int8_t) (sum);
}

/* I = Input layer, W = Weights, B = Biases, O = Output layer       */
/* idim/odim = size of input/output layers (i.e. number of neurons) */

static void nn_compute_layer(int8_t* I, int8_t* W, int8_t* B, int8_t* O, int idim, int odim, int8_t min, int8_t max) {
	#if defined(NN_WITH_AVX)
		const __m256i one = _mm256_set1_epi16(1);
		
		for (int o = 0; o < odim; o++) {
			__m256i sum = _mm256_setzero_si256();
			
			for (int i = 0; i < idim; i += 32) {
				const __m256i inp = _mm256_loadu_si256((__m256i*) &I[i]);
				const __m256i wei = _mm256_loadu_si256((__m256i*) &W[o * idim + i]);
				const __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(inp, wei), one);
				
				sum = _mm256_add_epi32(sum, dot);
			}
			
			const __m128i sum128lo = _mm256_castsi256_si128(sum);
			const __m128i sum128hi = _mm256_extracti128_si256(sum, 1);
			
			__m128i sum128 = _mm_add_epi32(sum128lo, sum128hi);
			
			sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_ABCD));
			sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
			
			O[o] = nn_clamp(_mm_cvtsi128_si32(sum128) + B[o] * FACTOR, min, max);
		}
	#else
		for (int o = 0; o < odim; o++) {
			int32_t sum = B[o] * FACTOR;
			
			// naive dot product
			
			for (int i = 0; i < idim; i++) {
				//sum += I[i] * W[i * odim + o];
				sum += I[i] * W[o * idim + i];
			}
			
			O[o] = nn_clamp(sum, min, max);
		}
	#endif
}


/****************************************************************************/
/** PUBLIC FUNCTIONS                                                       **/
/****************************************************************************/

int nn_convert(void) {
	printf("info debug NN file : conversion...\n");
	
	NN_Network* st = (NN_Network*) calloc(1, sizeof(NN_Network));
	
	if (st == NULL) {
		return -1;
	}
	
	FILE* file = fopen("network.txt", "r");
	
	if (file == NULL) {
		free(st);
		return -1;
	}
	
	/* load network (text format) */
	
	char line[256];
	
	// name
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "name=%[^\n]", &st->name) != 1) {
		fclose(file);
		free(st);
		return -1;
	}
	
	// author
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "author=%[^\n]", &st->author) != 1) {
		fclose(file);
		free(st);
		return -1;
	}
	
	// balance between win ratio and material
	
	float f = 0.0f;
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "wr=%f", &f) != 1) {
		fclose(file);
		free(st);
		return -1;
	}
	
	st->wr = f;
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "mt=%f", &f) != 1) {
		fclose(file);
		free(st);
		return -1;
	}
	
	st->mt = f;
	
	int8_t value = 0;
	
	// W0
	for (int row = 0; row < NN_SIZE_L1; row++) {
		for (int col = 0; col < 768; col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
				fclose(file);
				free(st);
				return -1;
			}
			//st->W0[row * 768 + col] = value;
			st->W0[col * NN_SIZE_L1 + row] = value; // transposition here !
		}
	}
	
	// B0
	for (int row = 0; row < NN_SIZE_L1; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
			fclose(file);
			free(st);
			return -1;
		}
		st->B0[row] = value;
	}
	
	// W1
	for (int row = 0; row < NN_SIZE_L2; row++) {
		for (int col = 0; col < (NN_SIZE_L1 * 2); col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
				fclose(file);
				free(st);
				return -1;
			}
			st->W1[row * (NN_SIZE_L1 * 2) + col] = value;
		}
	}
	
	// B1
	for (int row = 0; row < NN_SIZE_L2; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
			fclose(file);
			free(st);
			return -1;
		}
		st->B1[row] = value;
	}
	
	// W2
	for (int row = 0; row < NN_SIZE_L3; row++) {
		for (int col = 0; col < NN_SIZE_L2; col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
				fclose(file);
				free(st);
				return -1;
			}
			st->W2[row * NN_SIZE_L2 + col] = value;
		}
	}
	
	// B2
	for (int row = 0; row < NN_SIZE_L3; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
			fclose(file);
			free(st);
			return -1;
		}
		st->B2[row] = value;
	}
	
	// W3
	for (int row = 0; row < NN_SIZE_L4; row++) {
		for (int col = 0; col < NN_SIZE_L3; col++) {
				if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
				fclose(file);
				free(st);
				return -1;
			}
			st->W3[row * NN_SIZE_L3 + col] = value;
		}
	}
	
	// B3
	for (int row = 0; row < NN_SIZE_L4; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%d", &value) != 1) {
			fclose(file);
			free(st);
			return -1;
		}
		st->B3[row] = value;
	}
	
	fclose(file);
	
	/* save network (binary format) */
	
	file = fopen(NN_FILE, "wb");
	
	if (file == NULL) {
		free(st);
		return -1;
	}
	
	fwrite(st, sizeof(NN_Network), 1, file);
	
	fclose(file);
	free(st);
	
	return 0;
}

int nn_load(char* filename) {
	*nn = (NN_Network) {0};
	
	FILE* file = fopen(filename, "rb");
	
	if (file == NULL) {
		fclose(file);
		return -1;
	}
	
	fread(nn, sizeof(NN_Network), 1, file);
	
	fclose(file);
	
	printf("info debug NN infos : %s by %s\n", nn->name, nn->author);
	
	return 0;
}

void nn_add_piece(NN_Accumulator accumulator, int piece_type, int piece_color, int piece_position) {
	#if defined(NN_DEBUG)
		assert(piece_type >= 0 && piece_type < 6);
		assert(piece_color >= 0 && piece_color < 2);
		assert(piece_position >= 0 && piece_position < 64);
	#endif
	
	const int index_w = (piece_type << 1) + (piece_color);
	const int index_b = (piece_type << 1) + (1 - piece_color);
	
	#if defined(NN_DEBUG)
		assert(index_w >= 0 && index_w < 12);
		assert(index_b >= 0 && index_b < 12);
	#endif
	
	const int sq_w = piece_position;
	const int sq_b = piece_position ^ 56;
	
	const int feature_w = (64 * index_w) + (sq_w);
	const int feature_b = (64 * index_b) + (sq_b);
	
	#if defined(NN_DEBUG)
		assert(feature_w >= 0 && feature_w < 768);
		assert(feature_b >= 0 && feature_b < 768);
	#endif
	
	#if defined(NN_WITH_AVX)
		__m256i acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*) &accumulator[0][o]);
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_w * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*) &accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*) &accumulator[1][o]);
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_b * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*) &accumulator[1][o], acc);
		}
	#else
		for (int o = 0; o < NN_SIZE_L1; o++) {
			accumulator[0][o] += nn->W0[feature_w * NN_SIZE_L1 + o];
			accumulator[1][o] += nn->W0[feature_b * NN_SIZE_L1 + o];
		}
	#endif
}

void nn_del_piece(NN_Accumulator accumulator, int piece_type, int piece_color, int piece_position) {
	#if defined(NN_DEBUG)
		assert(piece_type >= 0 && piece_type < 6);
		assert(piece_color >= 0 && piece_color < 2);
		assert(piece_position >= 0 && piece_position < 64);
	#endif
	
	const int index_w = (piece_type << 1) + (piece_color);
	const int index_b = (piece_type << 1) + (1 - piece_color);
	
	#if defined(NN_DEBUG)
		assert(index_w >= 0 && index_w < 12);
		assert(index_b >= 0 && index_b < 12);
	#endif
	
	const int sq_w = piece_position;
	const int sq_b = piece_position ^ 56;
	
	const int feature_w = (64 * index_w) + (sq_w);
	const int feature_b = (64 * index_b) + (sq_b);
	
	#if defined(NN_DEBUG)
		assert(feature_w >= 0 && feature_w < 768);
		assert(feature_b >= 0 && feature_b < 768);
	#endif
	
	#if defined(NN_WITH_AVX)
		__m256i acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*) &accumulator[0][o]);
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_w * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*) &accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*) &accumulator[1][o]);
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_b * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*) &accumulator[1][o], acc);
		}
	#else
		for (int o = 0; o < NN_SIZE_L1; o++) {
			accumulator[0][o] -= nn->W0[feature_w * NN_SIZE_L1 + o];
			accumulator[1][o] -= nn->W0[feature_b * NN_SIZE_L1 + o];
		}
	#endif
}

void nn_mov_piece(NN_Accumulator accumulator, int piece_type, int piece_color, int from, int to) {
	#if defined(NN_DEBUG)
		assert(piece_type >= 0 && piece_type < 6);
		assert(piece_color >= 0 && piece_color < 2);
		assert(from >= 0 && from < 64 && to >= 0 && to < 64);
	#endif
	
	const int index_w = (piece_type << 1) + (piece_color);
	const int index_b = (piece_type << 1) + (1 - piece_color);
	
	#if defined(NN_DEBUG)
		assert(index_w >= 0 && index_w < 12);
		assert(index_b >= 0 && index_b < 12);
	#endif
	
	const int fr_w = from;
	const int fr_b = from ^ 56;
	
	const int to_w = to;
	const int to_b = to ^ 56;
	
	const int feature_w_fr = (64 * index_w) + (fr_w);
	const int feature_b_fr = (64 * index_b) + (fr_b);
	
	const int feature_w_to = (64 * index_w) + (to_w);
	const int feature_b_to = (64 * index_b) + (to_b);
	
	#if defined(NN_DEBUG)
		assert(feature_w_fr >= 0 && feature_w_fr < 768);
		assert(feature_b_fr >= 0 && feature_b_fr < 768);
		assert(feature_w_to >= 0 && feature_w_to < 768);
		assert(feature_b_to >= 0 && feature_b_to < 768);
	#endif
	
	#if defined(NN_WITH_AVX)
		__m256i acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*) &accumulator[0][o]);
			
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_w_fr * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_w_to * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			
			_mm256_storeu_si256((__m256i*) &accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*) &accumulator[1][o]);
			
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_b_fr * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			
			wei = _mm256_loadu_si256((__m256i*) &nn->W0[feature_b_to * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			
			_mm256_storeu_si256((__m256i*) &accumulator[1][o], acc);
		}
	#else
		for (int o = 0; o < NN_SIZE_L1; o++) {
			accumulator[0][o] -= nn->W0[feature_w_fr * NN_SIZE_L1 + o];
			accumulator[0][o] += nn->W0[feature_w_to * NN_SIZE_L1 + o];
			
			accumulator[1][o] -= nn->W0[feature_b_fr * NN_SIZE_L1 + o];
			accumulator[1][o] += nn->W0[feature_b_to * NN_SIZE_L1 + o];
		}
	#endif
}

void nn_update_all_pieces(NN_Accumulator accumulator, const uint64_t* whites, const uint64_t* blacks) {
	#if defined(NN_WITH_AVX)
		memcpy(&(accumulator[0]), &(nn->B0[0]), NN_SIZE_L1 * sizeof(int16_t));
		memcpy(&(accumulator[1]), &(nn->B0[0]), NN_SIZE_L1 * sizeof(int16_t));
	#else
		for (int o = 0; o < NN_SIZE_L1; o++) {
			accumulator[0][o] = nn->B0[o];
			accumulator[1][o] = nn->B0[o];
		}
	#endif
	
	for (int piece_color = 0; piece_color <= 1; piece_color++) {
		for (int piece_type = 0; piece_type <= 5; piece_type++) {
			uint64_t pieces = (piece_color ? blacks : whites)[piece_type];
			
			while (pieces) {
				const int piece_position = NN_GET_POSITION(pieces);
				nn_add_piece(accumulator, piece_type, piece_color, piece_position);
				NN_POP_POSITION(pieces);
			}
		}
	}
}

float nn_evaluate(NN_Accumulator accumulator, int color) {
	#if defined(NN_DEBUG)
		assert(color == 0 || color == 1);
	#endif
	
	// layer 1 (concatenation of accumulators)
	
	int8_t L1[NN_SIZE_L1 * 2];
	
	for (int o = 0; o < NN_SIZE_L1; o++) {
		L1[o             ] = nn_clamp_acc(accumulator[    color][o]);
		L1[o + NN_SIZE_L1] = nn_clamp_acc(accumulator[1 - color][o]);
	}
	
	// layers 2, 3 & 4
	
	int8_t L2[NN_SIZE_L2], L3[NN_SIZE_L3], L4[NN_SIZE_L4];
	
	nn_compute_layer(L1, nn->W1, nn->B1, L2, NN_SIZE_L1 * 2, NN_SIZE_L2,          0, THRESHOLD);
	nn_compute_layer(L2, nn->W2, nn->B2, L3, NN_SIZE_L2    , NN_SIZE_L3,          0, THRESHOLD);
	nn_compute_layer(L3, nn->W3, nn->B3, L4, NN_SIZE_L3    , NN_SIZE_L4, -THRESHOLD, THRESHOLD);
	
	// win ratio & material
	
	float wr = ((float) (L4[0])) / ((float) FACTOR);
	float mt = ((float) (L4[1])) / ((float) FACTOR);
	
	// combination of win ratio & material
	
	return (nn->wr * wr) + (nn->mt * mt);
}
