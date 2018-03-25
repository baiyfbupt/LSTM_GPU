#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <cstdlib>

#include "lstm.h"
#include "utilities.h"

lstm_model_t **model_layers;

int main(int argc, char *argv[])
{
	int i = 0, c, p = 0;
	size_t file_size = 0, sz = 0;
	float *X_test = (float*)malloc(sizeof(float) * NI * NK);
	if (X_test == NULL)
		return -1;
	printf("initialized x_test\n");
	for(int j = 0; j < NK; j++)//NI * NK 按列存储
		for(int i = 0; i < NI; i++){
			X_test[j * NI + i] = i + j;
		  // printf("X_test[%d,%d]:%f\t",i,j,X_test[j * NI + i]);
		}

	lstm_model_parameters_t params;
  srand( time ( NULL ) );

	int layers = LAYERS;
	params.layers = layers;

	model_layers = (lstm_model_t **)calloc(layers, sizeof(lstm_model_t*));

	if ( model_layers == NULL ) {
		printf("Error in init!\n");
		exit(-1);
	}

	p = 0;
	while ( p < layers ) {
		lstm_init_model(NI, NM, NC, NO, NR, NP, &model_layers[p],
			YES_FILL_IT_WITH_A_BUNCH_OF_RANDOM_NUMBERS_PLEASE, &params);
		++p;
	}
	printf("init model\n");

	if ( argc == 3 && !strcmp(argv[1], "-r") ) {
		lstm_read_net_layers(model_layers, argv[2]);
	}
	printf("loaded model\n");
	printf(
		"LSTM: %d Layers, Ni: %d, NM: %d, NK: %d,Nc: %d, No: %d, Np: %d, Nr: %d\n",
		 layers, NI, NM, NK, NC,NO, NP, NR);
	lstm_test(model_layers[0], model_layers, NK, X_test, layers);
	printf("LSTM inference Done!\n");

	free(model_layers);
	free(X_test);

	return 0;
}
