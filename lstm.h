#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MINIBATCH																						8
#define NK																									4096//input length

#define NI                                                  128
#define NM     																						  128
#define NC                                                  128
#define NO                                                  128
#define NP                                                  128
#define NR                                                  128

#define LAYERS                                              1
#define YES_FILL_IT_WITH_A_BUNCH_OF_ZEROS_PLEASE            1
#define YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE		0

#define STD_LOADABLE_NET_NAME									              "lstm_net.net"


typedef struct lstm_model_parameters_t {
	// For progress monitoring
	double loss_moving_avg;
	// For gradient descent
	double lambda;
	double softmax_temp;

	double beta1;
	double beta2;

	int layers;

	int model_regularize;

	// General parameters
	int mini_batch_size;
} lstm_model_parameters_t;

typedef struct lstm_model_t
{
		int ni;// Number of inputs
		int nm;// Number of vectors
		int nc;// Number of memory cells
		int no;// Number of outpus
		int np;// Number of non-recurrent prjection layer units
		int nr;// Number of recurrent projection layer units

		// Parameters
		lstm_model_parameters_t * params;

		//Weights
		float* Wix;//(nc x ni)
		float* Wir;//(nc x nr)
		float* Wic;//(nc x nc)
		float* Wfx;//(nc x ni)
		float* Wrf;//(nc x nr)
		float* Wcf;//(nc x nc)
		float* Wcx;//(nc x ni)
		float* Wcr;//(nc x nr)
		float* Wox;//(nc x ni)
		float* Wor;//(nc x nr)
		float* Woc;//(nc x nc)
		float* Wrm;//(nr x nc)
		float* Wpm;//(np x nc)
		float* Wyr;//(no x nr)
		float* Wyp;//(no x np)

		//bias
		float* bi;//(nc x 1)
		float* bf;//(nc x 1)
		float* bc;//(nc x 1)
		float* bo;//(nc x 1)
		float* by;//(no x 1)

} lstm_model_t;

typedef struct lstm_values_cache_t {
	//caches
	float* x;
	float* rt_old;
	float* ct_old;
	float* rt;
	float* ct;
	float* it;
	float* ft;
	float* ot;
	float* mt;
	float* pt;
	float* yt;
} lstm_values_cache_t;

typedef struct lstm_values_state_t {
	float* c;
	float* h;
} lstm_values_state_t;


void lstm_init_fail(const char * );
int lstm_init_model(int, int, int, int, int, int, lstm_model_t**,
	 int, lstm_model_parameters_t *);
void lstm_free_model(lstm_model_t*);
void lstm_cache_container_free(lstm_values_cache_t*);
lstm_values_cache_t*  lstm_cache_container_init(int,int, int, int, int, int);
void lstm_forward_propagate(lstm_model_t*, float *, lstm_values_cache_t*,
	 lstm_values_cache_t*);
void lstm_zero_the_model(lstm_model_t * );
void lstm_cache_container_set_start(lstm_values_cache_t * );
void lstm_read_net_layers(lstm_model_t**, const char *);
void lstm_test(lstm_model_t*, lstm_model_t**, int, float*, int);
#endif
