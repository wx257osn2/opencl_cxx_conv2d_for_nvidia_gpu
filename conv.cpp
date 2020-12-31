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

static std::string cu_get_error_name(::CUresult res){
  const char* ptr;
  const auto err = ::cuGetErrorName(res, &ptr);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cu_get_error_name" + cu_get_error_name(err));
  return std::string{ptr};
}

static void cu_init(){
  const auto err = ::cuInit(0);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cuInit: " + cu_get_error_name(err));
}

class cu_module{
  ::CUmodule data;
 public:
  explicit cu_module(const std::filesystem::path& path){
    const auto err = ::cuModuleLoad(&data, path.string().c_str());
    if(err != CUDA_SUCCESS)
      throw std::runtime_error("cuModuleLoadDataEx : " + cu_get_error_name(err));
  }
  ~cu_module(){::cuModuleUnload(data);}
  ::CUfunction get_function(const char* func_name)const{
    ::CUfunction func;
    const auto err = ::cuModuleGetFunction(&func, data, func_name);
    if(err != CUDA_SUCCESS)
      throw std::runtime_error("cuModuleGetFunction : " + cu_get_error_name(err));
    return func;
  }
};

::CUdevice cu_get_device(int ordinal){
  ::CUdevice dev;
  const auto err = ::cuDeviceGet(&dev, ordinal);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cuDeviceGet: " + cu_get_error_name(err));
  return dev;
}

struct cu_device_ptr{
  ::CUdeviceptr data;
  cu_device_ptr(std::size_t size){
    const auto err = ::cuMemAlloc(&data, size);
    if(err != CUDA_SUCCESS)
      throw std::runtime_error("cuMemAlloc: " + cu_get_error_name(err));
  }
  ~cu_device_ptr(){::cuMemFree(data);}
};

static void cu_memcpy_h2d(::CUdeviceptr dst, const void* src, std::size_t size){
  const auto err = ::cuMemcpyHtoD(dst, src, size);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cuMemcpyHtoD: " + cu_get_error_name(err));
}

static void cu_memcpy_d2h(void* dst, ::CUdeviceptr src, std::size_t size){
  const auto err = ::cuMemcpyDtoH(dst, src, size);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cuMemcpyDtoH: " + cu_get_error_name(err));
}

static void cu_fillzero(::CUdeviceptr dst, std::size_t size){
  const auto err = ::cuMemsetD8(dst, 0, size);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cuMemsetD8: " + cu_get_error_name(err));
}

class cu_context{
  ::CUcontext data;
 public:
  cu_context(CUdevice dev, unsigned int flags = 0u){
    const auto err = ::cuCtxCreate(&data, flags, dev);
    if(err != CUDA_SUCCESS)
      throw std::runtime_error("cuCtxCreate: " + cu_get_error_name(err));
  }
  ::CUcontext get()const noexcept{return data;}
  ~cu_context(){::cuCtxDestroy(data);}
  void set_current(){
    const auto err = ::cuCtxSetCurrent(data);
    if(err != CUDA_SUCCESS)
      throw std::runtime_error("cuCtxSetCurrent: " + cu_get_error_name(err));
  }
};

static void cu_launch_kernel(
    ::CUfunction f,
    void** kernel_params,
    unsigned int grid_dim_x,
    unsigned int grid_dim_y,
    unsigned int grid_dim_z,
    unsigned int block_dim_x,
    unsigned int block_dim_y,
    unsigned int block_dim_z,
    unsigned int shared_mem_bytes = 0,
    ::CUstream stream = nullptr,
    void** extra = nullptr){
  const auto err = ::cuLaunchKernel(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
                                    shared_mem_bytes, stream, kernel_params, extra);
  if(err != CUDA_SUCCESS)
    throw std::runtime_error("cuLaunchKernel: " + cu_get_error_name(err));
}

static void convolution_gpu(const std::filesystem::path& nvptx, image& im, const float* kernel, int kernel_size){
  assert(kernel_size % 2 == 1);
  std::vector<unsigned char> data(im.get_width()*im.get_height());
  if(im.get_channels() == 4)
    RGBA2R(im.get(), im.calc_size(), data.data());
  else if(im.get_channels() == 1)
    std::copy(im.get(), im.get() + im.calc_size(), data.begin());

  cu_init();
  auto dev = cu_get_device(0);
  cu_context ctx(dev);
  ctx.set_current();
  auto device_image = cu_device_ptr(data.size() * sizeof(unsigned char));
  auto device_output = cu_device_ptr(data.size() * sizeof(unsigned char));
  auto device_kernel = cu_device_ptr(kernel_size*kernel_size * sizeof(float));

  cu_memcpy_h2d(device_image.data, data.data(), data.size() * sizeof(unsigned char));
  cu_fillzero(device_output.data, data.size() * sizeof(unsigned char));
  cu_memcpy_h2d(device_kernel.data, kernel, kernel_size*kernel_size*sizeof(float));

  cu_module module(nvptx);
  auto func = module.get_function("convolution_general");

  {
    int width = im.get_width();
    int height = im.get_height();
    void* kernel_args[] = {
      &device_image.data,
      &width,
      &height,
      &device_kernel.data,
      &kernel_size,
      &device_output.data
    };
    cu_launch_kernel(func, kernel_args, 32, 32, 1, (width+31)/32, (height+31)/32, 1);
  }

  cu_memcpy_d2h(data.data(), device_output.data, data.size() * sizeof(unsigned char));

  if(im.get_channels() == 4)
    G2GGGA(data.data(), data.size(), im.get());
  else if(im.get_channels() == 1)
    std::copy(data.begin(), data.end(), im.get());
}

int main(int argc, char** argv)try{
  if(argc != 4)
    return EXIT_FAILURE;
  image im{argv[1]};
  constexpr std::size_t kernel_size = 3;
  constexpr std::size_t kernel_square = kernel_size * kernel_size;
  const std::array<float, kernel_square> kernel = {{
    -1.f, 0, 1.f,
    -2.f, 0, 2.f,
    -1.f, 0, 1.f
  }};
  convolution_gpu(argv[3], im, kernel.data(), kernel_size);
  im.save(argv[2]);
  return EXIT_SUCCESS;
}catch(std::exception& e){
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}
