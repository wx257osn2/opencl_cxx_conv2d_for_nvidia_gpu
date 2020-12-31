#include<memory>
#include<filesystem>
#include<vector>
#include<cassert>
#include<algorithm>
#include<ranges>
#include<stdexcept>
#include<string>
#include<iostream>

#include<cuda.h>
#include<cuda_runtime_api.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include<stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include<stb/stb_image_write.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

class image{
  int width;
  int height;
  int channels;
  std::unique_ptr<unsigned char, decltype(&::stbi_image_free)> pixels;
 public:
  [[nodiscard]] explicit image(const std::filesystem::path& path)
    : pixels{::stbi_load(path.string().c_str(), &this->width, &this->height, &this->channels, 0), &::stbi_image_free}{}
  [[nodiscard]] const unsigned char* get()const noexcept{return pixels.get();}
  [[nodiscard]] unsigned char* get()noexcept{return pixels.get();}
  [[nodiscard]] int get_width()noexcept{return width;}
  [[nodiscard]] int get_height()noexcept{return height;}
  [[nodiscard]] int get_channels()noexcept{return channels;}
  void save(const std::filesystem::path& path){
    ::stbi_write_png(path.string().c_str(), width, height, channels, get(), 0);
  }
  [[nodiscard]] std::size_t calc_size()const noexcept{
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels);
  }
};

inline void RGBA2R(const unsigned char* src, std::size_t num, unsigned char* dst){
  for(std::size_t i : std::views::iota(static_cast<std::size_t>(0), num/4))
    dst[i] = src[i*4];
}

inline void G2GGGA(const unsigned char* src, std::size_t num, unsigned char* dst){
  for(std::size_t i : std::views::iota(static_cast<std::size_t>(0), num)){
    dst[i*4+0] = src[i];
    dst[i*4+1] = src[i];
    dst[i*4+2] = src[i];
    dst[i*4+3] = 255;
  }
}

template<typename T>
[[nodiscard]] static std::unique_ptr<T, decltype(&::cudaFree)> create_cuda_buffer(std::size_t N){
  void* ptr = nullptr;
  {
    const auto ret = ::cudaMalloc(&ptr, N * sizeof(T));
    if(ret != cudaSuccess)
      throw std::runtime_error(std::string{"cudaMalloc: "} + ::cudaGetErrorName(ret));
  }
  return std::unique_ptr<T, decltype(&::cudaFree)>(static_cast<T*>(ptr), &::cudaFree);
}

static void cuda_memcpy_h2d(void* dst, const void* src, std::size_t size){
  const auto err = ::cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if(err != cudaSuccess)
    throw std::runtime_error(std::string{"cudaMemcpy: "} + ::cudaGetErrorName(err));
}

static void cuda_memcpy_d2h(void* dst, const void* src, std::size_t size){
  const auto err = ::cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess)
    throw std::runtime_error(std::string{"cudaMemcpy: "} + ::cudaGetErrorName(err));
}

static void cuda_fillzero(void* dst, std::size_t size){
  const auto err = ::cudaMemset(dst, 0, size);
  if(err != cudaSuccess)
    throw std::runtime_error(std::string{"cudaMemset: "} + ::cudaGetErrorName(err));
}

extern void launch_convolution_gpu(
    const unsigned char* __restrict__ im,
    int width,
    int height,
    const float* __restrict__ kernel,
    int kernel_size,
    unsigned char* __restrict__ output);

static void convolution_gpu(image& im, const float* kernel, int kernel_size){
  assert(kernel_size % 2 == 1);
  std::vector<unsigned char> data(im.get_width()*im.get_height());
  if(im.get_channels() == 4)
    RGBA2R(im.get(), im.calc_size(), data.data());
  else if(im.get_channels() == 1)
    std::copy(im.get(), im.get() + im.calc_size(), data.begin());

  auto device_image = create_cuda_buffer<unsigned char>(data.size());
  auto device_output = create_cuda_buffer<unsigned char>(data.size());
  auto device_kernel = create_cuda_buffer<float>(kernel_size*kernel_size);

  cuda_memcpy_h2d(device_image.get(), data.data(), data.size() * sizeof(unsigned char));
  cuda_fillzero(device_output.get(), data.size() * sizeof(unsigned char));
  cuda_memcpy_h2d(device_kernel.get(), kernel, kernel_size*kernel_size*sizeof(float));

  launch_convolution_gpu(
    device_image.get(),
    im.get_width(),
    im.get_height(),
    device_kernel.get(),
    kernel_size,
    device_output.get());

  cuda_memcpy_d2h(data.data(), device_output.get(), data.size() * sizeof(unsigned char));

  if(im.get_channels() == 4)
    G2GGGA(data.data(), data.size(), im.get());
  else if(im.get_channels() == 1)
    std::copy(data.begin(), data.end(), im.get());
}

int main(int argc, char** argv)try{
  if(argc != 3)
    return EXIT_FAILURE;
  image im{argv[1]};
  constexpr std::size_t kernel_size = 3;
  constexpr std::size_t kernel_square = kernel_size * kernel_size;
  const std::array<float, kernel_square> kernel = {{
    -1.f, 0, 1.f,
    -2.f, 0, 2.f,
    -1.f, 0, 1.f
  }};
  convolution_gpu(im, kernel.data(), kernel_size);
  im.save(argv[2]);
  return EXIT_SUCCESS;
}catch(std::exception& e){
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}
