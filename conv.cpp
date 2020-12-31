#include<memory>
#include<filesystem>
#include<vector>
#include<cassert>
#include<algorithm>
#include<ranges>

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

static void convolution_cpu(image& im, const float* kernel, int kernel_size){
  assert(kernel_size % 2 == 1);
  std::vector<unsigned char> data(im.get_width()*im.get_height());
  std::vector<unsigned char> output(im.get_width()*im.get_height());
  if(im.get_channels() == 4)
    RGBA2R(im.get(), im.calc_size(), data.data());
  else if(im.get_channels() == 1)
    std::copy(im.get(), im.get() + im.calc_size(), data.begin());

  const int half_k = kernel_size / 2;
  for(int y : std::views::iota(0, im.get_height()))
    for(int x : std::views::iota(0, im.get_width())){
      if(y < half_k
      || im.get_height() - half_k <= y
      || x < half_k
      || im.get_width() - half_k <= x)
        continue;
      float t = 0.f;
      for(int i : std::views::iota(0, kernel_size))
        for(int j : std::views::iota(0, kernel_size))
          t += data[(y+j-half_k)*im.get_width()+x+i-half_k] * kernel[i*kernel_size+j];
      output[y*im.get_width()+x] = static_cast<unsigned char>(std::clamp(t,
                                                                         static_cast<float>(std::numeric_limits<unsigned char>::min()),
                                                                         static_cast<float>(std::numeric_limits<unsigned char>::max())));
    }

  if(im.get_channels() == 4)
    G2GGGA(output.data(), output.size(), im.get());
  else if(im.get_channels() == 1)
    std::copy(output.begin(), output.end(), im.get());
}

int main(int argc, char** argv){
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
  convolution_cpu(im, kernel.data(), kernel_size);
  im.save(argv[2]);
}
