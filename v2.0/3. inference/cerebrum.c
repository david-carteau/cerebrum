/*
 * The Cerebrum library and engine
 * Copyright (c) 2020-2025, by David Carteau. All rights reserved.
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
/** AUTHOR: David Carteau, France, November 2025                           **/
/** LICENSE: MIT (see above and "license.txt" file content)                **/
/****************************************************************************/

/*
 * Simplified example of usage (to adapt to your engine):
 * 
 * Assuming that pieces are stored in bitboards with a1=0 and h8=63
 * Assuming for colors that WHITE=0, BLACK=1
 * Assuming	for pieces that PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
 * 
 * 1) when starting:
 * #include "cerebrum.h"
 * nn_load(NN_FILE);
 *
 * NN_Accumulator accumulator;
 * NN_Accumulator accumulator_copy;
 * 
 * const uint64_t* whites = &white.pieces[PAWN];
 * const uint64_t* blacks = &black.pieces[PAWN];
 * 
 * 2) when initialising a position:
 * nn_update_all_pieces(accumulator, whites, blacks);
 * 
 * 3) when making move:
 * memcpy(&accumulator_copy, &accumulator, sizeof(accumulator));
 * 
 * if (castling || promotion || capture_en_passant) {
 *     nn_update_all_pieces(accumulator, whites, blacks);
 * } else {
 *     nn_mov_piece(accumulator, type_of_moved_piece, color_of_moved_piece, from, to);
 *     
 *     if (captured) {
 *         nn_del_piece(accumulator, type_of_captured_piece, color_of_captured_piece, to);
 *     }
 * }
 * 
 * 4) when unmaking move:
 * memcpy(&accumulator, &accumulator_copy, sizeof(accumulator));
 * 
 * 5) when evaluating:
 * nn_evaluate(accumulator, color_of_side_to_move);
 * 
 * Important notes:
 * - evaluation returned is from the side-to-move point of view
 * - evaluation returned needs to be multiplied by 100 to be equivalent to centipawns
 */

#include "cerebrum.h"

#define NN_GET_POSITION(pieces) __builtin_ctzll(pieces)
#define NN_POP_POSITION(pieces) pieces &= pieces - 1

// last layer identification
#if (NN_SIZE_L3 == None) && (NN_SIZE_L4 == None)
	#define NN_LAST_LAYER L2
#endif

#if (NN_SIZE_L3 != None) && (NN_SIZE_L4 == None)
	#define NN_LAST_LAYER L3
#endif

#if (NN_SIZE_L3 != None) && (NN_SIZE_L4 != None)
	#define NN_LAST_LAYER L4
#endif


/****************************************************************************/
/** TYPES & VARIABLES                                                      **/
/****************************************************************************/

typedef struct {
	char name[256];
	char author[256];
	
	int16_t W0[NN_SIZE_L0 * NN_SIZE_L1];
	int16_t B0[NN_SIZE_L1];
	
	int8_t W1[NN_SIZE_L1 * 2 * NN_SIZE_L2];
	int8_t B1[NN_SIZE_L2];
	
	#if NN_SIZE_L3 != None
	int8_t W2[NN_SIZE_L2 * NN_SIZE_L3];
	int8_t B2[NN_SIZE_L3];
	#endif
	
	#if NN_SIZE_L4 != None
	int8_t W3[NN_SIZE_L3 * NN_SIZE_L4];
	int8_t B3[NN_SIZE_L4];
	#endif
} NN_Network;

static const int8_t FACTOR = 64;

static NN_Network network;
static NN_Network* nn = &network;


/****************************************************************************/
/** PRIVATE FUNCTIONS                                                      **/
/****************************************************************************/

static int8_t nn_clamp_acc(int16_t sum) {
	if (sum < 0) {
		return 0;
	}
	
	if (sum > 127) {
		return 127;
	}
	
	return (int8_t) (sum);
}

static int8_t nn_clamp_lay(int32_t sum) {
	sum /= FACTOR;
	
	if (sum < 0) {
		return 0;
	}
	
	if (sum > 127) {
		return 127;
	}
	
	return (int8_t) (sum);
}

/* I = Input layer, W = Weights, B = Biases, O = Output layer       */
/* idim/odim = size of input/output layers (i.e. number of neurons) */

static void nn_compute_layer(int8_t* I, int8_t* W, int8_t* B, int8_t* O8, int32_t* O32, int idim, int odim) {
	#if defined(NN_WITH_AVX)
		const __m256i one = _mm256_set1_epi16(1);
		
		for (int o = 0; o < odim; o++) {
			__m256i sum = _mm256_setzero_si256();
			
			for (int i = 0; i < idim; i += 32) {
				const __m256i inp = _mm256_loadu_si256((__m256i_u*) &I[i]);
				const __m256i wei = _mm256_loadu_si256((__m256i_u*) &W[o * idim + i]);
				const __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(inp, wei), one);
				
				sum = _mm256_add_epi32(sum, dot);
			}
			
			const __m128i sum128lo = _mm256_castsi256_si128(sum);
			const __m128i sum128hi = _mm256_extracti128_si256(sum, 1);
			
			__m128i sum128 = _mm_add_epi32(sum128lo, sum128hi);
			
			sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_ABCD));
			sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
			
			if (O8 != NULL) {
				O8[o] = nn_clamp_lay(_mm_cvtsi128_si32(sum128) + B[o] * FACTOR);
			} else {
				O32[o] = (_mm_cvtsi128_si32(sum128) + B[o] * FACTOR) / FACTOR;
			}
		}
	#else
		for (int o = 0; o < odim; o++) {
			int32_t sum = B[o] * FACTOR;
			
			// naive dot product
			
			for (int i = 0; i < idim; i++) {
				sum += I[i] * W[o * idim + i];
			}
			
			if (O8 != NULL) {
				O8[o] = nn_clamp_lay(sum);
			} else {
				O32[o] = sum / FACTOR;
			}
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
		printf("info debug NN file : ERROR while allocating memory\n");
		return -1;
	}
	
	FILE* file = fopen("network.txt", "r");
	
	if (file == NULL) {
		printf("info debug NN file : ERROR while opening 'network.txt'\n");
		free(st);
		return -1;
	}
	
	/* load network (text format) */
	
	char line[256];
	
	// name
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "name=%[^\n]", st->name) != 1) {
		printf("info debug NN file : ERROR while parsing 'name'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	// author
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "author=%[^\n]", st->author) != 1) {
		printf("info debug NN file : ERROR while parsing 'author'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	// number of parameters
	
	int32_t loaded = 0;
	int32_t expected = 0;
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "parameters=%d", &expected) != 1) {
		printf("info debug NN file : ERROR while parsing 'parameters'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	// weights and biases
	
	int8_t value = 0;
	
	// W0
	for (int row = 0; row < NN_SIZE_L1; row++) {
		for (int col = 0; col < NN_SIZE_L0; col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
				printf("info debug NN file : ERROR while parsing 'W0'\n");
				fclose(file);
				free(st);
				return -1;
			}
			//st->W0[row * NN_SIZE_L0 + col] = value;
			st->W0[col * NN_SIZE_L1 + row] = value; // transposition here !
			loaded++;
		}
	}
	
	// B0
	for (int row = 0; row < NN_SIZE_L1; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
			printf("info debug NN file : ERROR while parsing 'B0'\n");
			fclose(file);
			free(st);
			return -1;
		}
		st->B0[row] = value;
		loaded++;
	}
	
	// W1
	for (int row = 0; row < NN_SIZE_L2; row++) {
		for (int col = 0; col < (NN_SIZE_L1 * 2); col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
				printf("info debug NN file : ERROR while parsing 'W1'\n");
				fclose(file);
				free(st);
				return -1;
			}
			st->W1[row * (NN_SIZE_L1 * 2) + col] = value;
			loaded++;
		}
	}
	
	// B1
	for (int row = 0; row < NN_SIZE_L2; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
			printf("info debug NN file : ERROR while parsing 'B1'\n");
			fclose(file);
			free(st);
			return -1;
		}
		st->B1[row] = value;
		loaded++;
	}
	
	#if NN_SIZE_L3 != None
	// W2
	for (int row = 0; row < NN_SIZE_L3; row++) {
		for (int col = 0; col < NN_SIZE_L2; col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
				printf("info debug NN file : ERROR while parsing 'W2'\n");
				fclose(file);
				free(st);
				return -1;
			}
			st->W2[row * NN_SIZE_L2 + col] = value;
			loaded++;
		}
	}
	
	// B2
	for (int row = 0; row < NN_SIZE_L3; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
			printf("info debug NN file : ERROR while parsing 'B2'\n");
			fclose(file);
			free(st);
			return -1;
		}
		st->B2[row] = value;
		loaded++;
	}
	#endif
	
	#if NN_SIZE_L4 != None
	// W3
	for (int row = 0; row < NN_SIZE_L4; row++) {
		for (int col = 0; col < NN_SIZE_L3; col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
				printf("info debug NN file : ERROR while parsing 'W3'\n");
				fclose(file);
				free(st);
				return -1;
			}
			st->W3[row * NN_SIZE_L3 + col] = value;
			loaded++;
		}
	}
	
	// B3
	for (int row = 0; row < NN_SIZE_L4; row++) {
		if (fgets(line, 256, file) == NULL || sscanf(line, "%hhd", &value) != 1) {
			printf("info debug NN file : ERROR while parsing 'B3'\n");
			fclose(file);
			free(st);
			return -1;
		}
		st->B3[row] = value;
		loaded++;
	}
	#endif
	
	fclose(file);
	
	/* check number of parameters */
	
	printf("info debug NN file : %i loaded parameters\n", loaded);
	printf("info debug NN file : %i expected parameters\n", expected);
	
	if (loaded != expected) {
		printf("info debug NN file : ERROR 'loaded' and 'expected' values differ\n");
		free(st);
		return -1;
	}
	
	/* save network (binary format) */
	
	file = fopen(NN_FILE, "wb");
	
	if (file == NULL) {
		printf("info debug NN file : ERROR while saving network\n");
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
		return -1;
	}
	
	size_t read = fread(nn, sizeof(NN_Network), 1, file);
	
	fclose(file);
	
	if (read == 0) {
		return -1;
	}
	
	printf("info debug NN infos : %s by %s\n", nn->name, nn->author);
	
	return 0;
}

void nn_init_accumulator(NN_Accumulator accumulator) {
	#if defined(NN_WITH_AVX)
		memcpy(&(accumulator[0]), &(nn->B0[0]), NN_SIZE_L1 * sizeof(int16_t));
		memcpy(&(accumulator[1]), &(nn->B0[0]), NN_SIZE_L1 * sizeof(int16_t));
	#else
		for (int o = 0; o < NN_SIZE_L1; o++) {
			accumulator[0][o] = nn->B0[o];
			accumulator[1][o] = nn->B0[o];
		}
	#endif
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
		assert(feature_w >= 0 && feature_w < NN_SIZE_L0);
		assert(feature_b >= 0 && feature_b < NN_SIZE_L0);
	#endif
	
	#if defined(NN_WITH_AVX)
		__m256i acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*)(void*) &accumulator[0][o]);
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_w * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*)(void*) &accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*)(void*) &accumulator[1][o]);
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_b * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*)(void*) &accumulator[1][o], acc);
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
		assert(feature_w >= 0 && feature_w < NN_SIZE_L0);
		assert(feature_b >= 0 && feature_b < NN_SIZE_L0);
	#endif
	
	#if defined(NN_WITH_AVX)
		__m256i acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*)(void*) &accumulator[0][o]);
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_w * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*)(void*) &accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*)(void*) &accumulator[1][o]);
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_b * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			_mm256_storeu_si256((__m256i*)(void*) &accumulator[1][o], acc);
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
		assert(feature_w_fr >= 0 && feature_w_fr < NN_SIZE_L0);
		assert(feature_b_fr >= 0 && feature_b_fr < NN_SIZE_L0);
		assert(feature_w_to >= 0 && feature_w_to < NN_SIZE_L0);
		assert(feature_b_to >= 0 && feature_b_to < NN_SIZE_L0);
	#endif
	
	#if defined(NN_WITH_AVX)
		__m256i acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*)(void*) &accumulator[0][o]);
			
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_w_fr * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_w_to * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			
			_mm256_storeu_si256((__m256i*)(void*) &accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 16) {
			acc = _mm256_loadu_si256((__m256i*)(void*) &accumulator[1][o]);
			
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_b_fr * NN_SIZE_L1 + o]);
			acc = _mm256_sub_epi16(acc, wei);
			
			wei = _mm256_loadu_si256((__m256i*)(void*) &nn->W0[feature_b_to * NN_SIZE_L1 + o]);
			acc = _mm256_add_epi16(acc, wei);
			
			_mm256_storeu_si256((__m256i*)(void*) &accumulator[1][o], acc);
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
	nn_init_accumulator(accumulator);
	
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

int nn_evaluate(NN_Accumulator accumulator, int color) {
	#if defined(NN_DEBUG)
		assert(color == 0 || color == 1);
	#endif
	
	// layer 1 (concatenation of accumulators)
	
	int8_t L1[NN_SIZE_L1 * 2];
	
	for (int o = 0; o < NN_SIZE_L1; o++) {
		L1[o             ] = nn_clamp_acc(accumulator[    color][o]);
		L1[o + NN_SIZE_L1] = nn_clamp_acc(accumulator[1 - color][o]);
	}
	
	// layer 2
	
	#if NN_SIZE_L3 != None
		int8_t L2[NN_SIZE_L2];
		nn_compute_layer(L1, nn->W1, nn->B1, L2, NULL, NN_SIZE_L1 * 2, NN_SIZE_L2);
	#else
		int32_t L2[NN_SIZE_L2];
		nn_compute_layer(L1, nn->W1, nn->B1, NULL, L2, NN_SIZE_L1 * 2, NN_SIZE_L2);
	#endif
	
	// layer 3
	
	#if NN_SIZE_L3 != None
		#if NN_SIZE_L4 != None
			int8_t L3[NN_SIZE_L3];
			nn_compute_layer(L2, nn->W2, nn->B2, L3, NULL, NN_SIZE_L2, NN_SIZE_L3);
		#else
			int32_t L3[NN_SIZE_L3];
			nn_compute_layer(L2, nn->W2, nn->B2, NULL, L3, NN_SIZE_L2, NN_SIZE_L3);
		#endif
	#endif
	
	// layer 4
	
	#if NN_SIZE_L4 != None
		int32_t L4[NN_SIZE_L4];
		nn_compute_layer(L3, nn->W3, nn->B3, NULL, L4, NN_SIZE_L3, NN_SIZE_L4);
	#endif
	
	// final evaluation
	
	const float eval = ((float) (NN_LAST_LAYER[0])) / ((float) FACTOR);
	
	return (int) (100.0f * eval);
}
