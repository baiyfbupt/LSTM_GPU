#ifndef LSTM_UTILITIES_H
#define LSTM_UTILITIES_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

void opencl_initalizer();
void sigmoid_forward(float *, float *, int, int);
void tanh_forward(float *, float *, int, int);
// used on contigous vectors
//		A = A + B		A,		B,    l
void gemm(float *, float *, float *, int, int, int);
void 	vectors_add(float*, float*, int, int);
void 	vectors_substract(float*, float*, int);
void broadcast_vectors_add(float*, float*, int , int);
//		A = A * B		A,		B,    l
void 	vectors_multiply(float*, float*, float*, int, int);
//						 V to be set, Length
int 	init_zero_vector(float**, int);
int 	free_vector(float**);
//		A = B       A,		B,		length
void 	copy_vector(float*, float*, int);
float* 	get_zero_vector(int);
float* 	get_random_vector(int,int);
void 	vector_set_to_zero(float*, int);
float sample_normal(void);
float randn(float, float);
void 	vector_read(float *, int, FILE *);
#endif
