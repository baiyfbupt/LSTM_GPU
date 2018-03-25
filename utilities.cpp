#include<iostream>
#include<cstdlib>

#include"../include/CL/cl_help.h"
#include"../utils/utils.h"

#include "utilities.h"

void opencl_initalizer(){
  OpenCLInit("/data/local/test/utilities.cl");
}

void sigmoid_forward(float *out, float *in, int row, int col){
  cl_mem bufin, bufout;
  cl_event *eventlist = new cl_event[2];
  int l = row * col / 4;
  size_t global[2];

  float *input;
  float *output;
  cl_int ret;
  // printf("sigmoid_forward\n");
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("in[%d,%d]:%f\n",i,j,in[j*row+i]);
  //   }
  bufin = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");
  bufout = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret,"clCreateBuffer");

  input = (float*)clEnqueueMapBuffer(command_queue, bufin, CL_TRUE,
    CL_MAP_WRITE, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(input, in, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufin, input, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  ret = clFlush(command_queue);
  ret = clFlush(command_queue);

  kernel = clCreateKernel(program, "sigmoid_forward", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufout);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufin);
  // ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&l);

  global[0] = l;
  global[1] = 1;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0,
     NULL, &eventlist[0]);
  CheckErr(ret, "clEnqueueNDRangeKernel in base");


  output = (float*)clEnqueueMapBuffer(command_queue, bufout, CL_TRUE,
    CL_MAP_READ, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(out, output, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufout, output, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("out[%d,%d]:%f\n",i,j,out[j*row+i]);
  //   }
  // printf("\n");

  ret = clFinish(command_queue);
  CheckErr(ret, "clFinish in base");
  return;
}

void tanh_forward(float *out, float *in, int row, int col){
  cl_mem bufin, bufout;
  cl_event *eventlist = new cl_event[2];
  size_t global[2];

  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("in[%d,%d]:%f\n",i,j,in[j*row+i]);
  //   }

  int l = row * col / 4 ;
  float *input;
  float *output;

  cl_int ret;
  // printf("tanh_forward\n");
  bufin = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");
  bufout = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret,"clCreateBuffer");

  input = (float*)clEnqueueMapBuffer(command_queue, bufin, CL_TRUE,
    CL_MAP_WRITE, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(input, in, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufin, input, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  ret = clFlush(command_queue);
  ret = clFlush(command_queue);

  kernel = clCreateKernel(program, "tanh_forward", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufout);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufin);
  // ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&l);

  global[0] = l;
  global[1] = 1;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL,
    0, NULL, &eventlist[0]);
  CheckErr(ret, "clEnqueueNDRangeKernel in base");

  output = (float*)clEnqueueMapBuffer(command_queue, bufout, CL_TRUE,
    CL_MAP_READ, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(out, output, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufout, output, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("out[%d,%d]:%f\n",i,j,out[j*row+i]);
  //   }
  // printf("\n");

  ret = clFinish(command_queue);
  CheckErr(ret, "clFinish in base");
  // free(input);
  // free(output);
  return;
}

void gemm(float *c, float *a,float *b, int d1, int d2, int d3){
  cl_mem bufa,bufb,bufc;
  cl_event *eventlist = new cl_event[2];
  size_t global[2];

  float *aa ;
  float *bb ;
  float *cc ;
  //for debug
  float *cbase = (float*)malloc(d1 * d3 * sizeof(float));
  float sum;
  for (int j = 0; j < d3; j++){
    for (int i = 0; i < d1; i++){
      sum = 0;
      for (int k = 0; k < d2; k++){
        sum += a[i * d2 + k] * b[j * d2 + k];
      }
      cbase[j * d1 + i] = sum;
    }
  }
  //
  // for(int i = 0; i < d1; i++)
  //   for(int j = 0; j < d2; j++){
  //     printf("a[%d,%d]:%f\t", i, j, a[i*d2+j]);
  //   }
  // for(int j = 0; j < d3; j++)
  //   for(int i = 0; i < d2; i++){
  //     printf("b[%d,%d]:%f\t", i, j, b[j*d2+i]);
  //   }
  // printf("d1:%d, d2:%d, d3:%d\n", d1, d2, d3);
  cl_int ret;
  // printf("begin gemm\n");
  //w
  bufa = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    d1 * d2 * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");
  //x
  bufb = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    d2 * d3 * sizeof(float), NULL, &ret);
  CheckErr(ret,"clCreateBuffer");
  // for wx
  bufc = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
     d1 * d3 * sizeof(float), NULL, &ret);
  CheckErr(ret,"clCreateBuffer");

  aa = (float*)clEnqueueMapBuffer(command_queue, bufa, CL_TRUE, CL_MAP_WRITE,
     0, d1 * d2 * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  bb = (float*)clEnqueueMapBuffer(command_queue, bufb, CL_TRUE, CL_MAP_WRITE,
     0, d2 * d3 * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");

  memcpy(aa, a, d1*d2*sizeof(float));
  memcpy(bb, b, d2*d3*sizeof(float));

  ret = clEnqueueUnmapMemObject(command_queue, bufa, aa, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");
  ret = clEnqueueUnmapMemObject(command_queue, bufb, bb, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  ret = clFlush(command_queue);
  ret = clFlush(command_queue);

  kernel = clCreateKernel(program, "gemm", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufc);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufa);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufb);
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&d1);
  ret = clSetKernelArg(kernel, 4, sizeof(int), (void*)&d2);
  ret = clSetKernelArg(kernel, 5, sizeof(int), (void*)&d3);

  int gx = d1 / 2;
  int gy = d3 / 2;
  // gx += (gx % lx == 0 ? 0:1) * lx;
  // gy += (gy % ly == 0 ? 0:1) * ly;
  global[0] = gx;
  global[1] = gy;

  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0,
     NULL, &eventlist[0]);
  CheckErr(ret, "clEnqueueNDRangeKernel in base");

  cc = (float*)clEnqueueMapBuffer(command_queue, bufc, CL_TRUE, CL_MAP_READ, 0,
     d1 * d3 * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(c, cc, d1*d3*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufc, cc, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");
  //for debug
  // for(int j = 0; j < d3; j++)
  //   for(int i = 0; i < d1; i++){
  //     printf("c[%d,%d]:%f\t", i, j, c[j*d1+i]);
  //   }
  // printf("\n");
  // for(int j = 0; j < d3; j++)
  //   for(int i = 0; i < d1; i++){
  //     printf("cbase[%d,%d]:%f\t", i, j, cbase[j*d1+i]);
  //   }
  // printf("\n");
  // for(int j = 0; j < d3; j++)
  //   for(int i = 0; i < d1; i++){
  //     if(fabs(c[j*d1+i] - cbase[j*d1+i])>0.01)
  //     {
  //       printf("[%d,%d]error\t",i,j);
  //       printf("c[%d,%d]:%f\t", i, j, c[j*d1+i]);
  //       printf("cbase[%d,%d]:%f\t", i, j, cbase[j*d1+i]);
  //     }
  //     else
  //       printf("[%d,%d]correct!\t",i,j);
  //       printf("c[%d,%d]:%f\t", i, j, c[j*d1+i]);
  //       printf("cbase[%d,%d]:%f\t", i, j, cbase[j*d1+i]);
  //   }
  // printf("verified\n");

  ret = clFinish(command_queue);
  CheckErr(ret, "clFinish in base");

  return;
}

void vectors_add(float *out, float *in, int row, int col){
  cl_mem bufin, bufout;
  cl_event *eventlist = new cl_event[2];
  size_t global[2];

  int l = row * col / 4;

  float *input;
  float *output;
  cl_int ret;
  //for debug
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("in[%d,%d]:%f\n",i,j,in[j*row+i]);
  //   }
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("out[%d,%d]:%f\n",i,j,out[j*row+i]);
  //   }

  // printf("vectors_add begin\n");

  bufin = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");

  bufout = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret,"clCreateBuffer");

  input = (float*)clEnqueueMapBuffer(command_queue, bufin, CL_TRUE,
     CL_MAP_WRITE, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  output = (float*)clEnqueueMapBuffer(command_queue, bufout, CL_TRUE,
    CL_MAP_WRITE, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(input,in,row*col*sizeof(float));
  memcpy(output,out,row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufin, input, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");
  ret = clEnqueueUnmapMemObject(command_queue, bufout, output, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  ret = clFlush(command_queue);
  ret = clFlush(command_queue);

  kernel = clCreateKernel(program, "vectors_add", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufout);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufin);

  global[0] = l;
  global[1] = 1;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL,
     0, NULL, &eventlist[0]);
  CheckErr(ret, "clEnqueueNDRangeKernel in base");

  output = (float*)clEnqueueMapBuffer(command_queue, bufout, CL_TRUE,
    CL_MAP_READ, 0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(out, output, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufout, output, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");
  //for debug
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("out[%d,%d]:%f\n",i,j,out[j*row+i]);
  //   }

  ret = clFinish(command_queue);
  CheckErr(ret, "clFinish in base");
  return;
}

//a for matrix;b for vector
void broadcast_vectors_add(float *a, float *b, int row, int col){
  cl_mem bufa, bufb;
  cl_event *eventlist = new cl_event[2];
  size_t global[2];
  float *aa;
  float *bb;
  cl_int ret;
  //for debug
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("a[%d,%d]:%f\t",i,j,a[j*row+i]);
  //   }
  // for(int i = 0; i < row; i++)
  //     printf("b[%d]:%f\t",i,b[i]);

  bufa = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");
  bufb = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
    row * sizeof(float), NULL, &ret);
  CheckErr(ret,"clCreateBuffer");

  aa = (float*)clEnqueueMapBuffer(command_queue, bufa, CL_TRUE, CL_MAP_WRITE,
    0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  bb = (float*)clEnqueueMapBuffer(command_queue, bufb, CL_TRUE, CL_MAP_WRITE,
    0, row * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(aa, a, row*col*sizeof(float));
  memcpy(bb, b, row*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufa, aa, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");
  ret = clEnqueueUnmapMemObject(command_queue, bufb, bb, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  ret = clFlush(command_queue);
  ret = clFlush(command_queue);

  kernel = clCreateKernel(program, "broadcast_vectors_add", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufa);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufb);
  ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&row);
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&col);

  global[0] = row/4;
  global[1] = col;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL,
     0, NULL, &eventlist[0]);
  CheckErr(ret, "clEnqueueNDRangeKernel in base");

  aa = (float*)clEnqueueMapBuffer(command_queue, bufa, CL_TRUE, CL_MAP_READ,
    0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(a, aa, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufa, aa, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  //for debug
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("a[%d,%d]:%f\t",i,j,a[j*row+i]);
  //   }
  // printf("\n");

  ret = clFinish(command_queue);
  CheckErr(ret, "clFinish in base");
  return;
}

void vectors_multiply(float*c, float *a, float *b, int row, int col){
  cl_mem bufa, bufb, bufc;
  cl_event *eventlist = new cl_event[2];
  size_t global[2];
  int l = row * col / 4;
  // printf("vectors_multiply\n");
  float *aa;
  float *bb;
  float *cc;

  cl_int ret;

  //for debug
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("a[%d,%d]:%f\n",i,j,a[j*row+i]);
  //   }
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("b[%d,%d]:%f\n",i,j,b[j*row+i]);
  //   }

  bufa = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
     row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");
  bufb = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
     row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");
  bufc = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
     row * col * sizeof(float), NULL, &ret);
  CheckErr(ret, "clCreateBuffer");

  aa = (float*)clEnqueueMapBuffer(command_queue, bufa, CL_TRUE, CL_MAP_WRITE,
     0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  bb = (float*)clEnqueueMapBuffer(command_queue, bufb, CL_TRUE, CL_MAP_WRITE,
     0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(aa, a, row*col*sizeof(float));
  memcpy(bb, b, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufa, aa, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");
  ret = clEnqueueUnmapMemObject(command_queue, bufb, bb, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  ret = clFlush(command_queue);
  ret = clFlush(command_queue);

  kernel = clCreateKernel(program, "vectors_multiply", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufc);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufa);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufb);

  global[0] = l;
  global[1] = 1;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL,
    0, NULL, &eventlist[0]);
  CheckErr(ret, "clEnqueueNDRangeKernel in base");

  cc = (float*)clEnqueueMapBuffer(command_queue, bufc, CL_TRUE, CL_MAP_READ,
    0, row * col * sizeof(float), 0, NULL, NULL, &ret);
  CheckErr(ret, "clEnqueueMapBuffer");
  memcpy(c, cc, row*col*sizeof(float));
  ret = clEnqueueUnmapMemObject(command_queue, bufc, cc, 0, NULL, NULL);
  CheckErr(ret, "clEnqueueUnmapMemObject in base");

  //for debug
  // for(int j = 0; j < col; j++)
  //   for(int i = 0; i < row; i++){
  //     printf("c[%d,%d]:%f\n",i,j,c[j*row+i]);
  //   }
  ret = clFinish(command_queue);
  CheckErr(ret, "clFinish in base");

  return;
}


float* get_random_vector(int l, int r) {

	int tmp = 0;
	float *p;
	p = (float*)calloc(l, sizeof(float));
	if ( p == NULL )
		exit(0);

	while ( tmp < l ){
		p[tmp] = randn(0,1) / sqrt( r / 5 );
		++tmp;
	}

	return p;
}

int init_zero_vector(float** v, int l)
{
	int tmp = 0;
	*v = (float*)calloc(l, sizeof(float));
	if ( *v == NULL )
		return -1;

	while ( tmp < l ){
		(*v)[tmp] = 0.0;
		++tmp;
	}

	return 0;
}

float* get_zero_vector(int l)
{
	int tmp = 0;
	float *p;
	p = (float*)calloc(l, sizeof(float));
	if ( p == NULL )
		exit(0);

	while ( tmp < l ){
		p[tmp] = 0.0;
		++tmp;
	}
	return p;
}

int free_vector(float** v)
{
	free(*v);
	*v = NULL;
	return 0;
}

void copy_vector(float* a, float* b, int l)
{
	int tmp = 0;

	while ( tmp < l ){
		a[tmp] = b[tmp];
		++tmp;
	}
}
void vector_set_to_zero(float* V, int l)
{
	int tmp = 0;
	while ( tmp < l )
		V[tmp++] = 0.0;
}

void vector_read(float * v, int l, FILE * fp)
{
	int tmp = 0;
	size_t i = 0;
	float * p;
	float value;

	while ( tmp < l ) {
		i = 0; p = &value;
		while ( i < sizeof(float) ) {
			*((char *)p) = fgetc(fp);
			++i; ++p;
		}
		v[tmp] = value;
		++tmp;
	}
}


float randn(float mu, float sigma)
{
  float u1, u2, w, mult;
  static float X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (float) X2);
    }

  do
    {
      u1 = -1 + ((float) rand () / RAND_MAX) * 2;
      u2 = -1 + ((float) rand () / RAND_MAX) * 2;
      w = pow (u1, 2) + pow (u2, 2);
    }
  while (w >= 1 || w == 0);

  mult = sqrt ((-2 * log (w)) / w);
  X1 = u1 * mult;
  X2 = u2 * mult;

  call = !call;

  return (mu + sigma * (float) X1);
}
