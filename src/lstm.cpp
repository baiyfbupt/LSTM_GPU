#include<iostream>
#include<cstdlib>

#include "lstm.h"
#include "utilities.h"

void lstm_init_fail(const char * msg)
{
	printf("%s",msg);
	exit(-1);
}

int lstm_init_model(int ni,int nm,int nc,int no,int nr, int np, lstm_model_t**
	model_to_be_set, int zeros, lstm_model_parameters_t * params)
{
	lstm_model_t* lstm = (lstm_model_t*)calloc(1, sizeof(lstm_model_t));
	if ( lstm == NULL )
		exit(-1);

	lstm->ni = ni;
	lstm->nm = nm;
	lstm->nc = nc;
	lstm->no = no;
	lstm->nr = nr;
	lstm->np = np;

	lstm->params = params;

	if ( zeros ) {
		//Weights
		lstm->Wix = get_zero_vector(nc * ni);
		lstm->Wir = get_zero_vector(nc * nr);
		lstm->Wic = get_zero_vector(nc * nc);
		lstm->Wfx = get_zero_vector(nc * ni);
		lstm->Wrf = get_zero_vector(nc * nr);
		lstm->Wcf = get_zero_vector(nc * nc);
		lstm->Wcx = get_zero_vector(nc * ni);
		lstm->Wcr = get_zero_vector(nc * nr);
		lstm->Wox = get_zero_vector(nc * ni);
		lstm->Wor = get_zero_vector(nc * nr);
		lstm->Woc = get_zero_vector(nc * nc);
		lstm->Wrm = get_zero_vector(nr * nc);
		lstm->Wpm = get_zero_vector(np * nc);
		lstm->Wyr = get_zero_vector(no * nr);
		lstm->Wyp = get_zero_vector(no * np);
	} else {
		//Weights
		lstm->Wix = get_random_vector(nc * ni, ni);
		lstm->Wir = get_random_vector(nc * nr, nr);
		lstm->Wic = get_random_vector(nc * nc, nc);
		lstm->Wfx = get_random_vector(nc * ni, ni);
		lstm->Wrf = get_random_vector(nc * nr, nr);
		lstm->Wcf = get_random_vector(nc * nc, nc);
		lstm->Wcx = get_random_vector(nc * ni, ni);
		lstm->Wcr = get_random_vector(nc * nr, nr);
		lstm->Wox = get_random_vector(nc * ni, ni);
		lstm->Wor = get_random_vector(nc * nr, nr);
		lstm->Woc = get_random_vector(nc * nc, nc);
		lstm->Wrm = get_random_vector(nr * nc, nc);
		lstm->Wpm = get_random_vector(np * nc, nc);
		lstm->Wyr = get_random_vector(no * nr, nr);
		lstm->Wyp = get_random_vector(no * np, np);
	}

	lstm->bf = get_zero_vector(nc);
	lstm->bi = get_zero_vector(nc);
	lstm->bc = get_zero_vector(nc);
	lstm->bo = get_zero_vector(nc);
	lstm->by = get_zero_vector(no);

	*model_to_be_set = lstm;

	return 0;
}
//					 lstm model to be freed
void lstm_free_model(lstm_model_t* lstm)
{
	//Weights
	free_vector(&lstm->Wix);
	free_vector(&lstm->Wir);
	free_vector(&lstm->Wic);
	free_vector(&lstm->Wfx);
	free_vector(&lstm->Wrf);
	free_vector(&lstm->Wcf);
	free_vector(&lstm->Wcx);
	free_vector(&lstm->Wcr);
	free_vector(&lstm->Wox);
	free_vector(&lstm->Wor);
	free_vector(&lstm->Woc);
	free_vector(&lstm->Wrm);
	free_vector(&lstm->Wpm);
	free_vector(&lstm->Wyr);
	free_vector(&lstm->Wyp);
	//bias
	free_vector(&lstm->bf);
	free_vector(&lstm->bi);
	free_vector(&lstm->bc);
	free_vector(&lstm->bo);
	free_vector(&lstm->by);

	free(lstm);
}

void lstm_cache_container_free(lstm_values_cache_t* cache_to_be_freed)
{
	//caches
	free_vector(&(cache_to_be_freed)->x);
	free_vector(&(cache_to_be_freed)->rt_old);
	free_vector(&(cache_to_be_freed)->ct_old);
	free_vector(&(cache_to_be_freed)->rt);
	free_vector(&(cache_to_be_freed)->ct);
	free_vector(&(cache_to_be_freed)->it);
	free_vector(&(cache_to_be_freed)->ft);
	free_vector(&(cache_to_be_freed)->ot);
	free_vector(&(cache_to_be_freed)->mt);
	free_vector(&(cache_to_be_freed)->pt);
	free_vector(&(cache_to_be_freed)->yt);
}

lstm_values_cache_t*  lstm_cache_container_init(int ni,int nm, int nc, int no,
	int np, int nr)
{

	lstm_values_cache_t* cache =
	(lstm_values_cache_t*)calloc(1, sizeof(lstm_values_cache_t));

	if ( cache == NULL )
		exit(-1);

	cache->x = get_zero_vector(ni * nm);
	cache->rt_old = get_zero_vector(nr * nm);
	cache->ct_old = get_zero_vector(nc * nm);
	cache->rt = get_zero_vector(nr * nm);
	cache->ct = get_zero_vector(nc * nm);
	cache->it = get_zero_vector(nc * nm);
	cache->ft = get_zero_vector(nc * nm);
	cache->ot = get_zero_vector(nc * nm);
	cache->mt = get_zero_vector(nc * nm);
	cache->pt = get_zero_vector(np * nm);
	cache->yt = get_zero_vector(no * nm);

	return cache;
}

//model, input,  state and cache values, &probs, &state and cache values
void lstm_forward_propagate(lstm_model_t* model, float * input,
	 lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out)
{
	int ni, nc, no, np, nr, i = 0;
	int nm;
	float *rt_old, *ct_old;

	rt_old = cache_in->rt;
	ct_old = cache_in->ct;

	ni = model->ni;
	nm = model->nm;//vector number of x
	nc = model->nc;
	no = model->no;
	np = model->np;
	nr = model->nr;

	float tmp_c1[nc * nm];
	float tmp_c2[nc * nm];
	float tmp_c3[nc * nm];
	float tmp_chat[nc * nm];
	float tmp_o1[no * nm];
	float tmp_o2[no * nm];


	//for debug
	// float test[4];
	// for(int i=0; i < 4; i++)
	// 	test[i] = i;
	// for(int i=0; i < 200; i++)
	// {
	// 	sigmoid_forward(tmp_c1, input, ni, nm);
	// }
	//vectors_add(input, input, ni, nm);
	// tanh_forward(tmp_c1, input, ni, nm);
	// vectors_multiply(input,input, input, ni, nm);
	// broadcast_vectors_add(input, test, ni, nm);
	// gemm(tmp_c1, model->Wix, input, nc, ni, nm);

	copy_vector(cache_out->rt_old, cache_in->rt, nr * nm);
	copy_vector(cache_out->ct_old, cache_in->ct, nc * nm);
	copy_vector(cache_out->x, input, ni * nm);
// it = sigmoid(Wix * Xt + Wir * rt-1 + Wic * ct-1 + bi)
	gemm(tmp_c1, model->Wix, input, nc, ni, nm);
	gemm(tmp_c2, model->Wir, rt_old, nc, nr, nm);
	gemm(tmp_c3, model->Wic, ct_old, nc, nc, nm);
	vectors_add(tmp_c1, tmp_c2, nc, nm);
	broadcast_vectors_add(tmp_c3, model->bi, nc, nm);
	vectors_add(tmp_c1, tmp_c3, nc, nm);
	sigmoid_forward(cache_out->it, tmp_c1, nc, nm);

//	ft = sigmoid(Wfx * Xt + Wrf * rt-1 + Wcf * ct-1 + bf)
	gemm(tmp_c1, model->Wfx, input, nc, ni, nm);
	gemm(tmp_c2, model->Wrf, rt_old, nc, nr, nm);
	gemm(tmp_c3, model->Wcf, ct_old, nc, nc, nm);
	vectors_add(tmp_c1, tmp_c2, nc, nm);
	broadcast_vectors_add(tmp_c3, model->bf, nc, nm);
	vectors_add(tmp_c1, tmp_c3, nc, nm);
	sigmoid_forward(cache_out->ft, tmp_c1, nc, nm);

//	Ct = ft * ct-1 + it * tanh(Wcx * Xt + Wcr * rt-1 + bc)
	gemm(tmp_c1, model->Wcx, input, nc, ni, nm);
	gemm(tmp_c2, model->Wcr, rt_old, nc, nr, nm);
	vectors_add(tmp_c1, tmp_c2, nc, nm);
	broadcast_vectors_add(tmp_c1, model->bc, nc, nm);
	tanh_forward(tmp_chat, tmp_c1, nc, nm);//get the ~c

	vectors_multiply(tmp_c1, cache_out->ft, ct_old, nc, nm);
	vectors_multiply(tmp_c2, cache_out->it, tmp_chat, nc, nm);
	vectors_add(tmp_c1, tmp_c2, nc, nm);
	copy_vector(cache_out->ct, tmp_c1, nc * nm);

//	Ot = sigmoid(Wox * Xt + Wor * rt-1 + Woc * ct + bo)
	gemm(tmp_c1, model->Wox, input, nc, ni, nm);
	gemm(tmp_c2, model->Wor, rt_old, nc, nr, nm);
	gemm(tmp_c3, model->Woc, cache_out->ct, nc, nc, nm);
	vectors_add(tmp_c1, tmp_c2, nc, nm);
	broadcast_vectors_add(tmp_c3, model->bo, nc, nm);
	vectors_add(tmp_c1, tmp_c3, nc, nm);
	sigmoid_forward(cache_out->ot, tmp_c1, nc, nm);

//	mt = Ot * tanh(ct)
	tanh_forward(tmp_c1, cache_out->ct, nc, nm);
	vectors_multiply(cache_out->mt, tmp_c1, cache_out->ot, nc, nm);

//	rt = Wrm * mt
	gemm(cache_out->rt, model->Wrm, cache_out->mt, nr, nc, nm);

//	pt = Wpm * mt
	gemm(cache_out->pt, model->Wpm, cache_out->mt, np, nc, nm);

//	yt = Wyr * rt + Wyp * pt + by
	gemm(tmp_o1, model->Wyr, cache_out->rt, no, nr, nm);
	gemm(tmp_o2, model->Wyp, cache_out->pt, no, np, nm);
	vectors_add(tmp_o1, tmp_o2, no, nm);
	broadcast_vectors_add(tmp_o1, model->by, no, nm);
	copy_vector(cache_out->yt, tmp_o1, no * nm);
}

void lstm_zero_the_model(lstm_model_t * model)
{
	//Weights
	vector_set_to_zero(model->Wix, model->nc * model->ni);
	vector_set_to_zero(model->Wir, model->nc * model->nr);
	vector_set_to_zero(model->Wic, model->nc * model->nc);
	vector_set_to_zero(model->Wfx, model->nc * model->ni);
	vector_set_to_zero(model->Wrf, model->nc * model->nr);
	vector_set_to_zero(model->Wcf, model->nc * model->nc);
	vector_set_to_zero(model->Wcx, model->nc * model->ni);
	vector_set_to_zero(model->Wcr, model->nc * model->nr);
	vector_set_to_zero(model->Wox, model->nc * model->ni);
	vector_set_to_zero(model->Wor, model->nc * model->nr);
	vector_set_to_zero(model->Woc, model->nc * model->nc);
	vector_set_to_zero(model->Wrm, model->nr * model->nc);
	vector_set_to_zero(model->Wpm, model->np * model->nc);
	vector_set_to_zero(model->Wyr, model->no * model->nr);
	vector_set_to_zero(model->Wyp, model->no * model->np);
	//bias
	vector_set_to_zero(model->by, model->no);
	vector_set_to_zero(model->bi, model->nc);
	vector_set_to_zero(model->bc, model->nc);
	vector_set_to_zero(model->bf, model->nc);
	vector_set_to_zero(model->bo, model->nc);
}

void lstm_cache_container_set_start(lstm_values_cache_t * cache)
{
	// State variables set to zero
	vector_set_to_zero(cache->rt, NR);
	vector_set_to_zero(cache->ct, NC);

}

void lstm_read_net_layers(lstm_model_t** model, const char * filename)
{
	FILE * fp;
	int p = 0;

	fp = fopen(filename, "r");

	if ( fp == NULL ) {
		printf("Failed to open file: %s for reading.\n", filename);
		return;
	}

	while ( p < LAYERS ) {
		//Weights
		vector_read(model[p]->Wix, model[p]->nc * model[p]->ni, fp);
		vector_read(model[p]->Wir, model[p]->nc * model[p]->nr, fp);
		vector_read(model[p]->Wic, model[p]->nc * model[p]->nc, fp);
		vector_read(model[p]->Wfx, model[p]->nc * model[p]->ni, fp);
		vector_read(model[p]->Wrf, model[p]->nc * model[p]->nr, fp);
		vector_read(model[p]->Wcf, model[p]->nc * model[p]->nc, fp);
		vector_read(model[p]->Wcx, model[p]->nc * model[p]->ni, fp);
		vector_read(model[p]->Wcr, model[p]->nc * model[p]->nr, fp);
		vector_read(model[p]->Wox, model[p]->nc * model[p]->ni, fp);
		vector_read(model[p]->Wor, model[p]->nc * model[p]->nr, fp);
		vector_read(model[p]->Woc, model[p]->nc * model[p]->nc, fp);
		vector_read(model[p]->Wrm, model[p]->nr * model[p]->nc, fp);
		vector_read(model[p]->Wpm, model[p]->np * model[p]->nc, fp);
		vector_read(model[p]->Wyr, model[p]->no * model[p]->nr, fp);
		vector_read(model[p]->Wyp, model[p]->no * model[p]->np, fp);

		//bias
		vector_read(model[p]->by, model[p]->no, fp);
		vector_read(model[p]->bi, model[p]->nc, fp);
		vector_read(model[p]->bc, model[p]->nc, fp);
		vector_read(model[p]->bf, model[p]->nc, fp);
		vector_read(model[p]->bo, model[p]->nc, fp);

		++p;
	}

	printf("Loaded the net: %s\n", filename);
	fclose(fp);
}

void lstm_test(lstm_model_t* model, lstm_model_t** model_layers, int len,
	float* X_test, int layers)
{
	int ni,nm,nc,no,np,nr,p;
	unsigned int i =0, q= 0, b=0, e1 = 0, e2 = 0, times, batch;//e1，e2

	unsigned int n = 0;

	// for(int i = 0; i < NI; i++)
	// 	for(int j = 0; j < NM; j++){
	// 		printf("lstm_test:X_test[%d,%d]:%f\t",i,j,X_test[i*NM+j]);
	// 	}

	lstm_values_cache_t ***cache_layers;

	ni = model->ni;
	nm = model->nm;
	nc = model->nc;
	no = model->no;
	np = model->np;
	nr = model->nr;

	// init the cache_layers
	i = 0;
	cache_layers = (lstm_values_cache_t***)calloc(layers,
		sizeof(lstm_values_cache_t**));
	if ( cache_layers == NULL )
		lstm_init_fail("Failed to allocate memory for the caches\n");

	while ( i < layers ) {
		cache_layers[i] = (lstm_values_cache_t**)calloc(MINIBATCH + 1,
			sizeof(lstm_values_cache_t*));
		if ( cache_layers[i] == NULL )
			lstm_init_fail("Failed to allocate memory for the caches\n");

		p = 0;//init cache layers
		while ( p < MINIBATCH + 1){
			cache_layers[i][p] = lstm_cache_container_init(ni, nm, nc, no, np, nr);
			if ( cache_layers[i][p] == NULL )
				lstm_init_fail("Failed to allocate memory for the caches\n");
			++p;
		}
		++i;
	}
	opencl_initalizer();

//start to test
	clock_t begin,end,t1,t2;
  begin = clock();
	times = len / (nm * MINIBATCH);
	if(len%(nm * MINIBATCH))
		times++;
	printf("len:%d\n",len);
	printf("times:%d\n",times);

	i = 0; b = 0;
	while (n < times){
		// b = i;
		//initialize every cell's states
		q = 0;
		while (q < layers) {
			lstm_cache_container_set_start(cache_layers[q][0]);
			++q;
		}

		//determine the length of input sequences
		batch = MINIBATCH;
		if ( i + MINIBATCH * nm >= len ) {
			batch = (len - i) / nm;
		}

			/* Layer numbering starts at the output point of the net */
		q = 0;
		printf("batch:%d\n",batch);
		while(q < batch){
			e1 = q;
			e2 = q + 1;

			p = layers - 1;
			lstm_forward_propagate(model_layers[p], X_test, cache_layers[p][e1],
				cache_layers[p][e2]);
			if ( p > 0 ) {
				--p;
				while ( p >= 0 ) {
					lstm_forward_propagate(model_layers[p], cache_layers[p+1][e2]->yt,
						cache_layers[p][e1], cache_layers[p][e2]);
					--p;
				}
				p = 0;
			}
			++q;
			i += nm ;
			// for debug
			// for(int j = 0; j < NM; j++)
			// 	for(int i = 0; i < NI; i++){
			// 		printf("X_test[%d,%d]:%f\t",i,j,X_test[j*NI+i]);
			// 	}
			// printf("\n");
			// printf("\n");
			X_test = X_test + nm * ni;
		}//while(q<trailing)
		++n;
	}//while( n < times )

	end = clock();
	double total = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	double speed = total / len;
	printf("%d vector(s) cost %f ms\n",len,total);
	printf("%f ms per vector\n",speed);

	p = 0;
	while ( p < layers ) {
		i = 0;
		while ( i < MINIBATCH + 1) {
			lstm_cache_container_free(cache_layers[p][i]);
			lstm_cache_container_free(cache_layers[p][i]);
			++i;
		}
		++p;
	}
	free(cache_layers);
}
