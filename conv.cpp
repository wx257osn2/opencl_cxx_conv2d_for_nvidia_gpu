#include<memory>
#include<filesystem>

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
};



int main(int argc, char** argv){
  if(argc != 3)
    return EXIT_FAILURE;
  image im{argv[1]};
  im.save(argv[2]);
}
