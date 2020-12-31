#include<cassert>

#include<cuda.h>

__global__ void convolution_general(
    const unsigned char* __restrict__ im,
    int width,
    int height,
    const float* __restrict__ kernel,
    int kernel_size,
    unsigned char* __restrict__ output){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int half_k = kernel_size / 2;
  if(y < half_k
  || height - half_k <= y
  || x < half_k
  || width - half_k <= x)
    return;
  float t = 0.f;
  for(int i = 0; i < kernel_size; ++i)
    for(int j = 0; j < kernel_size; ++j)
      t += im[(y+j-half_k)*width+x+i-half_k] * kernel[i*kernel_size+j];
  output[y*width+x] = static_cast<unsigned char>(min(max(t, 0.f), 255.f));
}

void launch_convolution_gpu(
    const unsigned char* __restrict__ im,
    int width,
    int height,
    const float* __restrict__ kernel,
    int kernel_size,
    unsigned char* __restrict__ output){
  assert(kernel_size % 2 == 1);
  const dim3 threads(32, 32);
  convolution_general<<<threads, dim3((width+threads.x-1)/threads.x, (height+threads.y-1)/threads.y)>>>(
    im, width, height, kernel, kernel_size, output);
}