import numpy as np
import pickle

out_js_header = """export const [n_diffusion, n_s, n_p, n_t] = [%d, %d, %d, %d];
export const cube_name_list = %s;
export const color_mode_list = %s;
export const cube_idx_list = %s;
export const n_cube = %d;
export const n_mode = %d;
export const lut_cube_bundle = new Uint16Array([%s]);
"""

path_list = [
  "cube_9x16x16x16_rgb_rgb_q384.pickle",
  "cube_9x16x16x16_rgb_yuv_q384.pickle",
  "cube_9x16x16x16_cmyk_rgb_q384.pickle",
  "cube_9x16x16x16_cmyk_yuv_q384.pickle",
  "cube_9x16x16x16_yuv_yuv_q384.pickle",
]

name_list = [
  "RGB_RGB",
  "RGB_YUV",
  "CMYK_RGB",
  "CMYK_YUV",
  "YUV_YUV",
  "RGB_LINEAR",
]

color_mode_list = [
  0, 1, 0, 1, 1, 2,
]

cube_idx_list = [
  0, 1, 2, 3, 4, 0,
]

assert len(color_mode_list) == len(cube_idx_list)

n_s, n_p, n_t = 16, 16, 16
n_diffusion = 9
out = np.zeros((len(path_list), n_diffusion + 1, n_s, n_p, n_t, 3), dtype = np.float16)

# add lut with no diffusion
for i_cube in range(len(path_list)):
  cube = out[i_cube, -1]
  for s in range(n_s):
    for p in range(n_p):
      for t in range(n_t):
        cube[s, p, t] = (s / (n_s - 1), p / (n_p - 1), t / (n_t - 1))

for i, path in enumerate(path_list):
  cube = pickle.load(open(path, "rb"))
  out[i, :-1] = cube.astype(np.float16)

data = ",".join(map(lambda x:hex(x), out.view(dtype = np.uint16).flatten()))
out_js = out_js_header % (data, n_diffusion + 1, n_s, n_p, n_t, len(path_list), str(name_list), str(color_mode_list), str(cube_idx_list), len(color_mode_list))

open("./cube_bundle.bin", "wb").write(out.tobytes())
open("./cube-bundle.js", "wb").write((out_js).encode("utf-8"))