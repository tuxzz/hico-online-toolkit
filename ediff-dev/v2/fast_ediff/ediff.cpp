#include <algorithm>
#include <random>
#include <array>
#include <limits>
#include <omp.h>
#include <ctime>

#if defined(__GNUC__)
#define DLL_EXPORT  __attribute__ ((dllexport))
#elif defined(_MSC_VER)
#define DLL_EXPORT __declspec(dllexport)
#endif

/* random */
uint32_t xorshift32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

class XorShift32 {
public:
  using result_type = uint32_t;

  inline XorShift32() : XorShift32(0x52391314) {}
  inline XorShift32(uint32_t seed) : m_seed(seed) {}

  inline void seed(uint32_t new_seed) { m_seed = new_seed; }

  inline uint32_t operator()() {
    m_seed = xorshift32(m_seed);
    return m_seed;
  }

  static constexpr uint32_t min() { return 1u; }
  static constexpr uint32_t max() { return 0xffffffffu - 1; }

private:
  uint32_t m_seed;
};

static thread_local XorShift32 g_random_generator;

/* CXX generic */
template<typename T, size_t N>
static inline std::array<T, N> ptr_to_array(const T *__restrict ptr) {
  std::array<T, N> arr;
  std::copy(ptr, ptr + N, arr.begin());
  return std::move(arr);
}

template<size_t C>
static inline void do_error_diffusion_pattern_mse_generic(std::array<float, C> * __restrict io_img, const size_t h, const size_t w, const std::array<float, C> * __restrict kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, std::array<float, C> * __restrict shuffle_pattern_list, const size_t n_pattern, std::array<float, C> * __restrict diffused_error_temp) {
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      std::shuffle(shuffle_pattern_list, shuffle_pattern_list + n_pattern, g_random_generator);
      const auto pix_value = io_img[y * w + x];

      std::array<float, C> best_pattern_signed_error;
      {
        size_t best_pattern_idx;
        float best_pattern_abs_error = std::numeric_limits<float>::max();
        for (size_t i_pattern = 0; i_pattern < n_pattern; ++i_pattern) {
          std::array<float, C> pattern_signed_error;
          float abs_error = 0.0;
          for (size_t i_c = 0; i_c < C; ++i_c) {
            const float err = shuffle_pattern_list[i_pattern][i_c] - pix_value[i_c];
            pattern_signed_error[i_c] = err;
            abs_error += std::abs(err);
          }
          if (abs_error < best_pattern_abs_error) {
            best_pattern_idx = i_pattern;
            best_pattern_signed_error = pattern_signed_error;
            best_pattern_abs_error = abs_error;
          }
        }
        io_img[y * w + x] = shuffle_pattern_list[best_pattern_idx];
      }
      
      for (size_t i_krnl = 0; i_krnl < h_krnl * w_krnl; ++i_krnl) {
        for (size_t i_c = 0; i_c < C; ++i_c)
          diffused_error_temp[i_krnl][i_c] = kernel[i_krnl][i_c] * best_pattern_signed_error[i_c];
      }

      const auto ib_y = std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(y) - static_cast<std::ptrdiff_t>(c_h_krnl));
      const auto ib_x = std::max<std::ptrdiff_t>(0, static_cast<std::ptrdiff_t>(x) - static_cast<std::ptrdiff_t>(c_w_krnl));
      const auto ie_y = std::min(h, y + h_krnl - c_h_krnl);
      const auto ie_x = std::min(w, x + w_krnl - c_w_krnl);
      const auto kb_y = ib_y - (y - c_h_krnl);
      const auto kb_x = ib_x - (x - c_w_krnl);
      const auto n_y = ie_y - ib_y;
      const auto n_x = ie_x - ib_x;
      for (size_t i_y = 0; i_y < n_y; ++i_y) {
        const auto y_krnl = kb_y + i_y;
        const auto y_img = ib_y + i_y;
        for (size_t i_x = 0; i_x < n_x; ++i_x) {
          const auto x_krnl = kb_x + i_x;
          const auto x_img = ib_x + i_x;
          for (size_t i_c = 0; i_c < C; ++i_c) {
            io_img[y_img * w + x_img][i_c] -= diffused_error_temp[y_krnl * w_krnl + x_krnl][i_c];
          }
        }
      }
    }
  }
}

template<size_t C>
static inline std::array<float, C> convert_color_generic(std::array<float, C> *io_color_blk, const size_t blk_size, const std::array<float, C> *kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, std::array<float, C> *shuffle_pattern_list, size_t n_pattern, std::array<float, C> *diffused_error_temp) {
  do_error_diffusion_pattern_mse_generic<C>(io_color_blk, blk_size, blk_size, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, shuffle_pattern_list, n_pattern, diffused_error_temp);

  std::array<float, C> out = { 0.0f };

  auto blk_size_2 = blk_size * blk_size;
  for (size_t i = 0; i < blk_size_2; ++i) {
    for (size_t i_c = 0; i_c < C; ++i_c)
      out[i_c] += io_color_blk[i][i_c];
  }

  auto blk_size_2_f = static_cast<float>(blk_size_2);
  for (size_t i_c = 0; i_c < C; ++i_c)
    out[i_c] /= blk_size_2_f;

  return out;
}

template<size_t C>
static inline void convert_many_color_inplace_generic_mt(std::array<float, C> *io_color_list, const size_t n_color, const std::array<float, C> *kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, const std::array<float, C> *pattern_list, size_t n_pattern, const size_t blk_size, const float randomness) {
  const auto single_buffer_size = blk_size * blk_size;
  const auto n_max_thread = static_cast<size_t>(omp_get_max_threads());
  const auto t = static_cast<uint32_t>(time(nullptr));
  std::vector<std::array<float, C>> buffer(single_buffer_size * n_max_thread);
  std::vector<std::array<float, C>> pattern_buffer(n_pattern * n_max_thread);
  std::vector<std::array<float, C>> diffused_error_temp(h_krnl * w_krnl * n_max_thread);
  for (auto i_thread = 0; i_thread < n_max_thread; ++i_thread)
    std::copy(pattern_list, pattern_list + n_pattern, pattern_buffer.data() + i_thread * n_pattern);
#pragma omp parallel for 
  for (int i_color = 0; i_color < n_color; ++i_color) {
    const auto current_thread_id = static_cast<size_t>(omp_get_thread_num());
    auto my_buffer = buffer.data() + single_buffer_size * current_thread_id;
    auto my_pattern = pattern_buffer.data() + n_pattern * current_thread_id;
    auto my_diffuse_error_temp = diffused_error_temp.data() + h_krnl * w_krnl * current_thread_id;
    g_random_generator.seed(t ^ static_cast<uint32_t>(current_thread_id));

    auto color = io_color_list[i_color];
    std::fill(my_buffer, my_buffer + blk_size * blk_size, color);
    if (randomness > 0.0f)
    {
      std::uniform_real_distribution<float> rnd(-randomness, randomness);
      for (size_t i = 0; i < single_buffer_size; ++i) {
        for (size_t i_c = 0; i_c < C; ++i_c)
          my_buffer[i][i_c] += rnd(g_random_generator);
      }
    }
    io_color_list[i_color] = convert_color_generic<C>(my_buffer, blk_size, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, my_pattern, n_pattern, my_diffuse_error_temp);
  }
}

/* C Export */
extern "C" {
  DLL_EXPORT void do_error_diffusion_pattern_mse_c3(std::array<float, 3> *io_img, const size_t h, const size_t w, const std::array<float, 3> *kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, std::array<float, 3> *shuffle_pattern_list, const size_t n_pattern, std::array<float, 3> *diffused_error_temp) {
    do_error_diffusion_pattern_mse_generic<3>(io_img, h, w, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, shuffle_pattern_list, n_pattern, diffused_error_temp);
  }

  DLL_EXPORT void convert_many_color_inplace_mt_c3(std::array<float, 3> *io_color_list, const size_t n_color, const std::array<float, 3> *kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, const std::array<float, 3> *pattern_list, size_t n_pattern, const size_t blk_size, const float randomness) {
    convert_many_color_inplace_generic_mt<3>(io_color_list, n_color, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, blk_size, randomness);
  }

  DLL_EXPORT void do_error_diffusion_pattern_mse_c4(std::array<float, 4> *io_img, const size_t h, const size_t w, const std::array<float, 4> *kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, std::array<float, 4> *shuffle_pattern_list, const size_t n_pattern, std::array<float, 4> *diffused_error_temp) {
    do_error_diffusion_pattern_mse_generic<4>(io_img, h, w, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, shuffle_pattern_list, n_pattern, diffused_error_temp);
  }

  DLL_EXPORT void convert_many_color_inplace_mt_c4(std::array<float, 4> *io_color_list, const size_t n_color, const std::array<float, 4> *kernel, const size_t h_krnl, const size_t w_krnl, const size_t c_h_krnl, const size_t c_w_krnl, const std::array<float, 4> *pattern_list, size_t n_pattern, const size_t blk_size, const float randomness) {
    convert_many_color_inplace_generic_mt<4>(io_color_list, n_color, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, blk_size, randomness);
  }
}