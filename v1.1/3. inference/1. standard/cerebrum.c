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
/** NAME: cerebrum.c (inference)                                           **/
/** AUTHOR: David Carteau, France, February 2025                           **/
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
 * nn_evaluate(accumulator, (color == WHITE ? 0 : 1));
 * 
 * Important notes about evaluation:
 * - evaluation returned is from the side-to-move point of view
 * - evaluation returned needs to be multiplied by 1000 to be equivalent to centipawns
 * 
 * To get the best possible inference speed:
 * - define the C macro 'NN_WITH_AVX' (e.g. '-D NN_WITH_AVX' flag for clang/gcc)
 * - make sure that the AVX and AVX2 instruction sets are enabled ('-mavx -mavx2' flags for clang/gcc)
 * - consider also using the C macro 'NN_WITH_FMA' and the FMA instruction set (variable results: test!)
 */

#include "cerebrum.h"

#define NN_GET_POSITION(pieces) __builtin_ctzll(pieces)
#define NN_POP_POSITION(pieces) pieces &= pieces - 1

// last layer identification & minimum clamp value for each layer
#if (NN_SIZE_L3 == None) && (NN_SIZE_L4 == None)
	#define NN_LAST_LAYER L2
	#define NN_CLAMP_MINIMUM_L2 -Q
#endif

#if (NN_SIZE_L3 != None) && (NN_SIZE_L4 == None)
	#define NN_LAST_LAYER L3
	#define NN_CLAMP_MINIMUM_L2 0
	#define NN_CLAMP_MINIMUM_L3 -Q
#endif

#if (NN_SIZE_L3 != None) && (NN_SIZE_L4 != None)
	#define NN_LAST_LAYER L4
	#define NN_CLAMP_MINIMUM_L2 0
	#define NN_CLAMP_MINIMUM_L3 0
	#define NN_CLAMP_MINIMUM_L4 -Q
#endif


/****************************************************************************/
/** TYPES & VARIABLES                                                      **/
/****************************************************************************/

typedef struct {
	char name[256];
	char author[256];
	
	float wr;
	float mt;
	
	float W0[NN_SIZE_L0 * NN_SIZE_L1];
	float B0[NN_SIZE_L1];
	
	float W1[NN_SIZE_L1 * 2 * NN_SIZE_L2];
	float B1[NN_SIZE_L2];
	
	#if NN_SIZE_L3 != None
	float W2[NN_SIZE_L2 * NN_SIZE_L3];
	float B2[NN_SIZE_L3];
	#endif
	
	#if NN_SIZE_L4 != None
	float W3[NN_SIZE_L3 * NN_SIZE_L4];
	float B3[NN_SIZE_L4];
	#endif
} NN_Network;

static const float Q = 127.0f / 64.0f;

static NN_Network network;
static NN_Network* nn = &network;


/****************************************************************************/
/** PRIVATE FUNCTIONS                                                      **/
/****************************************************************************/

static float nn_clamp_acc(float sum) {
	if (sum < 0.0f) {
		return 0.0f;
	}
	
	if (sum > Q) {
		return Q;
	}
	
	return sum;
}

static float nn_clamp_lay(float sum, float minimum) {
	if (sum < minimum) {
		return minimum;
	}
	
	if (sum > Q) {
		return Q;
	}
	
	return sum;
}

/* I = Input layer, W = Weights, B = Biases, O = Output layer       */
/* idim/odim = size of input/output layers (i.e. number of neurons) */

static void nn_compute_layer(float* I, float* W, float* B, float* O, int idim, int odim, float minimum) {
	for (int o = 0; o < odim; o++) {
		float sum = B[o];
		
		#if defined(NN_WITH_FMA)
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
			
			// intrinsics dot product (more complex, but performance gain)
			// see: https://stackoverflow.com/questions/59494745/avx2-computing-dot-product-of-512-float-arrays
			
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
				sum += I[i] * W[o * idim + i];
			}
		#endif
		
		O[o] = nn_clamp_lay(sum, minimum);
	}
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
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "name=%[^\n]", &st->name) != 1) {
		printf("info debug NN file : ERROR while parsing 'name'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	// author
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "author=%[^\n]", &st->author) != 1) {
		printf("info debug NN file : ERROR while parsing 'author'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	// balance between win ratio and material
	
	float f = 0.0f;
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "wr=%f", &f) != 1) {
		printf("info debug NN file : ERROR while parsing 'wr'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	st->wr = f;
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "mt=%f", &f) != 1) {
		printf("info debug NN file : ERROR while parsing 'mt'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	st->mt = f;
	
	// number of parameters
	
	int32_t loaded = 0;
	int32_t expected = 0;
	
	if (fgets(line, 256, file) == NULL || sscanf(line, "parameters=%d", &expected) != 1) {
		printf("info debug NN file : ERROR while parsing 'parameters'\n");
		fclose(file);
		free(st);
		return -1;
	}
	
	float value = 0.0f;
	
	// W0
	for (int row = 0; row < NN_SIZE_L1; row++) {
		for (int col = 0; col < NN_SIZE_L0; col++) {
			if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
		if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
			if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
		if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
			if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
		if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
			if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
		if (fgets(line, 256, file) == NULL || sscanf(line, "%f", &value) != 1) {
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
		memcpy(&(accumulator[0]), &(nn->B0[0]), NN_SIZE_L1 * sizeof(float));
		memcpy(&(accumulator[1]), &(nn->B0[0]), NN_SIZE_L1 * sizeof(float));
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
		__m256 acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 8) {
			acc = _mm256_loadu_ps(&accumulator[0][o]);
			wei = _mm256_loadu_ps(&nn->W0[feature_w * NN_SIZE_L1 + o]);
			acc = _mm256_add_ps(acc, wei);
			_mm256_storeu_ps(&accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 8) {
			acc = _mm256_loadu_ps(&accumulator[1][o]);
			wei = _mm256_loadu_ps(&nn->W0[feature_b * NN_SIZE_L1 + o]);
			acc = _mm256_add_ps(acc, wei);
			_mm256_storeu_ps(&accumulator[1][o], acc);
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
		__m256 acc, wei;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 8) {
			acc = _mm256_loadu_ps(&accumulator[0][o]);
			wei = _mm256_loadu_ps(&nn->W0[feature_w * NN_SIZE_L1 + o]);
			acc = _mm256_sub_ps(acc, wei);
			_mm256_storeu_ps(&accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 8) {
			acc = _mm256_loadu_ps(&accumulator[1][o]);
			wei = _mm256_loadu_ps(&nn->W0[feature_b * NN_SIZE_L1 + o]);
			acc = _mm256_sub_ps(acc, wei);
			_mm256_storeu_ps(&accumulator[1][o], acc);
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
		__m256 acc;
		
		// white's pov
		for (int o = 0; o < NN_SIZE_L1; o += 8) {
			acc = _mm256_loadu_ps(&accumulator[0][o]);
			acc = _mm256_sub_ps(acc, _mm256_loadu_ps(&nn->W0[feature_w_fr * NN_SIZE_L1 + o]));
			acc = _mm256_add_ps(acc, _mm256_loadu_ps(&nn->W0[feature_w_to * NN_SIZE_L1 + o]));
			_mm256_storeu_ps(&accumulator[0][o], acc);
		}
		
		// black's pov
		for (int o = 0; o < NN_SIZE_L1; o += 8) {
			acc = _mm256_loadu_ps(&accumulator[1][o]);
			acc = _mm256_sub_ps(acc, _mm256_loadu_ps(&nn->W0[feature_b_fr * NN_SIZE_L1 + o]));
			acc = _mm256_add_ps(acc, _mm256_loadu_ps(&nn->W0[feature_b_to * NN_SIZE_L1 + o]));
			_mm256_storeu_ps(&accumulator[1][o], acc);
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

float nn_evaluate(NN_Accumulator accumulator, int color) {
	#if defined(NN_DEBUG)
		assert(color == 0 || color == 1);
	#endif
	
	// layer 1 (concatenation of accumulators)
	
	float L1[NN_SIZE_L1 * 2];
	
	for (int o = 0; o < NN_SIZE_L1; o++) {
		L1[o             ] = nn_clamp_acc(accumulator[    color][o]);
		L1[o + NN_SIZE_L1] = nn_clamp_acc(accumulator[1 - color][o]);
	}
	
	// layer 2
	
	float L2[NN_SIZE_L2];
	
	nn_compute_layer(L1, nn->W1, nn->B1, L2, NN_SIZE_L1 * 2, NN_SIZE_L2, NN_CLAMP_MINIMUM_L2);
	
	// other layers, if any
	
	#if NN_SIZE_L3 != None
	float L3[NN_SIZE_L3];
	
	nn_compute_layer(L2, nn->W2, nn->B2, L3, NN_SIZE_L2, NN_SIZE_L3, NN_CLAMP_MINIMUM_L3);
	#endif
	
	#if NN_SIZE_L4 != None
	float L4[NN_SIZE_L4];
	
	nn_compute_layer(L3, nn->W3, nn->B3, L4, NN_SIZE_L3, NN_SIZE_L4, NN_CLAMP_MINIMUM_L4);
	#endif
	
	// win ratio & material
	
	const float wr = NN_LAST_LAYER[0];
	const float mt = NN_LAST_LAYER[1];
	
	// combination of win ratio & material
	
	return (nn->wr * wr) + (nn->mt * mt);
}
