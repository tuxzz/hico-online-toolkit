import numpy as np
import cv2
import ediff
import pickle

prefix = "yuv_yuv"
randomness = 0.1
np_dtype = np.float32
diffuse_fac = np.linspace(0.0, 1.0, 9, dtype = np.float32)
cube_shape = (diffuse_fac.shape[0], 16, 16, 16)
c = len(cube_shape) - 1
quality = 384

'''
pattern_list = np.array((
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
), dtype = np_dtype)
'''
'''
pattern_list = np.array((
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
), dtype = np_dtype)
'''

pattern_list = np.array((
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 0.5, 0.5),
    (1, 0.5, 0.5),
), dtype = np_dtype)

#pattern_list = ediff.sbgr_to_yuv(pattern_list)
print(prefix, quality)
print(pattern_list)

kernel = np.array((
  (0, 0, 0, 7, 5),
  (3, 5, 7, 5, 3),
  (1, 3, 5, 3, 1),
), dtype = np_dtype)
c_h_krnl = 0
c_w_krnl = 2

kernel /= np.sum(kernel)
kernel = kernel.repeat(c).reshape(*kernel.shape, c)


print("Generate Input Cube...")
n_diffuse_fac, n_s, n_p, n_t = cube_shape
cube_input = np.empty((*cube_shape[1:], c), dtype = np.float32)
for s in range(n_s):
    for p in range(n_p):
      for t in range(n_t):
        cube_input[s, p, t] = (s / (n_s - 1), p / (n_p - 1), t / (n_t - 1))
cube_input = cube_input.reshape(n_s * n_p * n_t, c)

print("Generate LUT Cube...")
cube = np.empty((*cube_shape, 3), dtype = np.float32)
for i_diffuse_fac, diffuse_fac in enumerate(diffuse_fac):
  print("Begin... %dx%dx%d = %d, diffuse = %f" % (n_s, n_p, n_t, n_s * n_p * n_t, diffuse_fac))
  cube[i_diffuse_fac] = ediff.convert_many_color_inplace(cube_input, kernel * diffuse_fac, c_h_krnl, c_w_krnl, pattern_list, quality, randomness).reshape(n_s, n_p, n_t, c)

with open("out_cube_%dx%dx%dx%d_%s_q%d.pickle" % (n_diffuse_fac, n_s, n_p, n_t, prefix, quality), "wb") as f:
  pickle.dump(cube, f)
print("Done!")