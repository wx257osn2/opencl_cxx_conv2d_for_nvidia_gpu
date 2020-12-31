__kernel void convolution_general(
    __global const unsigned char* __restrict__ im,
    int width,
    int height,
    __global const float* __restrict__ kern,
    int kernel_size,
    __global unsigned char* __restrict__ output){
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int half_k = kernel_size / 2;
  if(y < half_k
  || height - half_k <= y
  || x < half_k
  || width - half_k <= x)
    return;
  float t = 0.f;
  for(int i = 0; i < kernel_size; ++i)
    for(int j = 0; j < kernel_size; ++j)
      t += im[(y+j-half_k)*width+x+i-half_k] * kern[i*kernel_size+j];
  output[y*width+x] = (unsigned char)(min(max(t, 0.f), 255.f));
}
