import numpy as np
import numba as nb
import pylab as pl
import cv2
import ediff
from mpl_toolkits.mplot3d import Axes3D

p_low, p_high = -0.1, 0.1
size = 128
np_dtype = np.float32

pattern_list = np.array((
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
), dtype = np_dtype)
'''
kernel = np.array((
  (0, 0, 7),
  (3, 5, 1),
), dtype = np_dtype)
c_h_krnl = 0
c_w_krnl = 1
kernel /= np.sum(kernel)
'''
kernel = np.array((
  (0, 0, 0, 7, 5),
  (3, 5, 7, 5, 3),
  (1, 3, 5, 3, 1),
), dtype = np_dtype)
kernel /= np.sum(kernel)
c_h_krnl = 0
c_w_krnl = 2

kernel *= 0.5
print("Load Image...")
h, w, c = size, size, 3
assert c == 3

print("Generate noise_tex...")
#t_filter = sp.firwin(9, 0.5 - 1e-5, window = "nuttall", pass_zero = False, scale = True, nyq = 1.0)
noise_tex = np.random.uniform(low = p_low, high = p_high, size = (h, w, c)).astype(np_dtype)
#p_tex = conv_2d(p_tex, t_filter)

print("Add noise...")
#img += noise_tex

print("Do...")
kernel = kernel.repeat(3).reshape(*kernel.shape, 3)

@nb.njit(parallel=True, fastmath=True)
def do():
  out = np.zeros((256 // 4, 256 // 4, len(pattern_list)), dtype = np_dtype)
  for ii in nb.prange(256 // 4):
    i = ii * 4
    for jj in nb.prange(256 // 4):
      j = jj * 4
      v = i / 255
      vv = j / 255
      iv = np.array([v, vv, 0], dtype = np_dtype)
      blk = np.empty((size, size, 3), dtype = np_dtype)
      blk[:] = (iv)
      print(i)
      img_pattern_idx = ediff.do_error_diffusion_pattern_mse_pidx(blk, kernel, c_h_krnl, c_w_krnl, pattern_list)
      for i_pattern in range(len(pattern_list)):
        out[ii, jj, i_pattern] = np.mean(img_pattern_idx == i_pattern)
  return out
out = do()

X, Y = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4))
for i, color in enumerate(pattern_list):
  fig = pl.figure()
  ax = fig.gca(projection = '3d')
  pl.title(str(color))
  ax.plot_surface(X, Y, out[:, :, i])
#pl.legend()
pl.show()