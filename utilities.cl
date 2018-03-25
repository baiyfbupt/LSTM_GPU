#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_arm_printf : enable


__kernel void gemm(
  __global  float *c,
  const __global float4 *a,
  const __global float *b,
  const int d1,
  const int d2,
  const int d3
)
{
  const int row = get_global_id(0) * 2;
  const int col = get_global_id(1) * 2;

  float4 c00 = 0.0f;
  float4 c01 = 0.0f;
  float4 a0, a1;
  float4 b0;
  float2 temp;
  int end = d2 / 4;
  for (int i = 0; i < end; i++){
    vstore4(a[row * end + i], 0, (float*)&a0);
    vstore4(a[(row+1) * end + i], 0, (float*)&a1);
    b0 = vload4(0, b + col * d2 + i * 4);

    c00 += a0 * b0;
    c01 += a1 * b0;
  }
  c00.s0 += c00.s1 + c00.s2 + c00.s3 ;
  c01.s0 += c01.s1 + c01.s2 + c01.s3 ;
  temp = (float2)(c00.s0, c01.s0);
  vstore2(temp, 0, c + col * d1 + row);

  c00 = 0.0f;
  c01 = 0.0f;
  for (int i = 0; i < end; i++){
    vstore4(a[(row) * end + i], 0, (float*)&a0);
    vstore4(a[(row + 1) * end + i], 0, (float*)&a1);
    b0 = vload4(0, b + (col + 1) * d2 + i * 4);

    c00 += a0 * b0;
    c01 += a1 * b0;
  }
  c00.s0 += c00.s1 + c00.s2 + c00.s3 ;
  c01.s0 += c01.s1 + c01.s2 + c01.s3 ;
  temp = (float2)(c00.s0, c01.s0);
  vstore2(temp, 0, c + (col + 1) * d1 + row);
}


__kernel void sigmoid_forward(
  __global float *out,
  const __global float *in
)
{
    const int x = get_global_id(0);

    float4 a0;
    a0 = vload4(x, in);
    a0.s0 = 1 / (1 + native_exp(-a0.s0));
    a0.s1 = 1 / (1 + native_exp(-a0.s1));
    a0.s2 = 1 / (1 + native_exp(-a0.s2));
    a0.s3 = 1 / (1 + native_exp(-a0.s3));
    vstore4(a0, x, out);
}

__kernel void tanh_forward(
  __global float *out,
  const __global float *in
)
{
    const int x = get_global_id(0);

    float4 a0;
    a0 = vload4(x, in);
    a0.s0 = tanh(a0.s0);
    a0.s1 = tanh(a0.s1);
    a0.s2 = tanh(a0.s2);
    a0.s3 = tanh(a0.s3);
    vstore4(a0, x, out);
}

__kernel void vectors_add(
  __global float *out,
  const __global float *in
)
{
    const int x = get_global_id(0);

    float4 a0, a1;
    a0 = vload4(x, out);
    a1 = vload4(x, in);
    a0 = a0 + a1;
    vstore4(a0, x, out);
}

__kernel void vectors_multiply(
  __global float *c,
  const __global float *a,
  const __global float *b
)
{
    const int x = get_global_id(0);

    float4 a0, a1;
    a0 = vload4(x, a);
    a1 = vload4(x, b);
    a0 = a0 * a1;
    vstore4(a0, x, c);
}

__kernel void broadcast_vectors_add(
  __global float *a,
  const __global float *b,
  const int row,
  const int col
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  float4 a0, b0;
  a0 = vload4(x, a + y * row);
  b0 = vload4(x, b);
  a0 = a0 + b0;
  vstore4(a0, x, a + y * row);
}
