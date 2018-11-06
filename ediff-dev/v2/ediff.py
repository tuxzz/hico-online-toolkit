import numpy as np
import numba as nb
import sys, ctypes
import numpy.ctypeslib as npct

_array_1d = npct.ndpointer(dtype = np.float32, ndim = 1, flags = "C")
_array_2d = npct.ndpointer(dtype = np.float32, ndim = 2, flags = "C")
_array_3d = npct.ndpointer(dtype = np.float32, ndim = 3, flags = "C")
_size_t = ctypes.c_size_t

_dll = ctypes.CDLL("fast_ediff.dll")

# void (io_img, h, w, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, shuffle_pattern_list, n_pattern, diffused_error_temp)
_do_error_diffusion_pattern_mse_c3 = _dll.do_error_diffusion_pattern_mse_c3
_do_error_diffusion_pattern_mse_c3.restype = None
_do_error_diffusion_pattern_mse_c3.argtypes = [_array_3d, _size_t, _size_t, _array_3d, _size_t, _size_t, _size_t, _size_t, _array_2d, _size_t, _array_3d]

#void (io_color_list, n_color, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, blk_size, randomness)
_convert_many_color_inplace_mt_c3 = _dll.convert_many_color_inplace_mt_c3
_convert_many_color_inplace_mt_c3.restype = None
_convert_many_color_inplace_mt_c3.argtypes = [_array_2d, _size_t, _array_3d, _size_t, _size_t, _size_t, _size_t, _array_2d, _size_t, _size_t, ctypes.c_float]

# void (io_img, h, w, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, shuffle_pattern_list, n_pattern, diffused_error_temp)
_do_error_diffusion_pattern_mse_c4 = _dll.do_error_diffusion_pattern_mse_c4
_do_error_diffusion_pattern_mse_c4.restype = None
_do_error_diffusion_pattern_mse_c4.argtypes = [_array_3d, _size_t, _size_t, _array_3d, _size_t, _size_t, _size_t, _size_t, _array_2d, _size_t, _array_3d]

#void (io_color_list, n_color, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, blk_size, randomness)
_convert_many_color_inplace_mt_c4 = _dll.convert_many_color_inplace_mt_c4
_convert_many_color_inplace_mt_c4.restype = None
_convert_many_color_inplace_mt_c4.argtypes = [_array_2d, _size_t, _array_3d, _size_t, _size_t, _size_t, _size_t, _array_2d, _size_t, _size_t, ctypes.c_float]

@nb.njit(cache=True, nogil=True, fastmath=True)
def do_error_diffusion_pattern_mse_py(img, kernel, c_h_krnl, c_w_krnl, pattern_list):
  h, w, c = img.shape
  h_krnl, w_krnl, c_krnl = kernel.shape
  n_pattern, c_pattern = pattern_list.shape
  assert h_krnl > 0 and w_krnl > 0
  assert h_krnl > c_h_krnl >= 0 and w_krnl > c_w_krnl >= 0
  assert c_pattern == c
  assert c_krnl == c
  assert n_pattern > 0

  if c_h_krnl > 0:
    assert (kernel[0:c_h_krnl - 1, 0:c_w_krnl] == 0.0).all()
  assert (kernel[c_h_krnl, 0:c_w_krnl] == 0.0).all()

  img = img.copy()
  pattern_order_list = np.arange(n_pattern)
  for y in range(h):
    for x in range(w):
      np.random.shuffle(pattern_order_list)
      pattern_list = pattern_list[pattern_order_list, :]
      pix_value = img[y, x]
      error_list = pattern_list - pix_value.reshape(1, 3)
      best_pattern_idx = np.argmin(np.sum(np.abs(error_list), axis = 1))
      img[y, x] = pattern_list[best_pattern_idx]
      best_error = error_list[best_pattern_idx]
      diffused_error = kernel.reshape(h_krnl, w_krnl, 3) * best_error.reshape(1, 1, 3)

      ib_y, ib_x = max(0, y - c_h_krnl), max(0, x - c_w_krnl)
      ie_y, ie_x = min(h, y + h_krnl - c_h_krnl), min(w, x + w_krnl - c_w_krnl)
      kb_y, kb_x = ib_y - (y - c_h_krnl), ib_x - (x - c_w_krnl)
      ke_y, ke_x = kb_y + ie_y - ib_y, kb_x + ie_x - ib_x
      #print(h, w, y, x, "/", ib_y, ie_y, ib_x, ie_x, "/", kb_y, ke_y, kb_x, ke_x)
      img[ib_y:ie_y, ib_x:ie_x] -= diffused_error[kb_y:ke_y, kb_x:ke_x]
  return img


def do_error_diffusion_pattern_mse(img, kernel, c_h_krnl, c_w_krnl, pattern_list):
  h, w, c = img.shape
  h_krnl, w_krnl, c_krnl = kernel.shape
  n_pattern, c_pattern = pattern_list.shape
  assert h_krnl > 0 and w_krnl > 0
  assert h_krnl > c_h_krnl >= 0 and w_krnl > c_w_krnl >= 0
  assert c_pattern == c
  assert c_krnl == c
  assert n_pattern > 0
  if c_h_krnl > 0:
    assert (kernel[0:c_h_krnl - 1, 0:c_w_krnl] == 0.0).all()
  assert (kernel[c_h_krnl, 0:c_w_krnl] == 0.0).all()

  if c in (3, 4):
    io_img = np.require(img.copy(), dtype = np.float32, requirements = "C")
    pattern_list = np.require(pattern_list.copy(), dtype = np.float32, requirements = "C")
    kernel = np.require(kernel, dtype = np.float32, requirements = "C")
    diffused_error_temp = np.empty_like(kernel, dtype = np.float32, order = "C")
    if c == 3:
      _do_error_diffusion_pattern_mse_c3(io_img, h, w, kernel, h_krnl ,w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, diffused_error_temp)
    elif c == 4:
      _do_error_diffusion_pattern_mse_c4(io_img, h, w, kernel, h_krnl ,w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, diffused_error_temp)
    return io_img
  else:
    return do_error_diffusion_pattern_mse_py(img, kernel, c_h_krnl, c_w_krnl, pattern_list)

@nb.njit(cache=True, nogil=True, fastmath=True)
def do_error_diffusion_pattern_mse_pidx(img, kernel, c_h_krnl, c_w_krnl, pattern_list):
  h, w, c = img.shape
  h_krnl, w_krnl, c_krnl = kernel.shape
  n_pattern, c_pattern = pattern_list.shape
  assert h_krnl > 0 and w_krnl > 0
  assert h_krnl > c_h_krnl >= 0 and w_krnl > c_w_krnl >= 0
  assert c_pattern == c
  assert c_krnl == c
  assert n_pattern > 0

  if c_h_krnl > 0:
    assert (kernel[0:c_h_krnl - 1, 0:c_w_krnl] == 0.0).all()
  assert (kernel[c_h_krnl, 0:c_w_krnl] == 0.0).all()

  img = img.copy()
  out = np.zeros((h, w), dtype = np.int32)
  pattern_order_list = np.arange(n_pattern)
  for y in range(h):
    for x in range(w):
      np.random.shuffle(pattern_order_list)
      local_pattern_list = pattern_list[pattern_order_list, :]
      pix_value = img[y, x]
      error_list = local_pattern_list - pix_value.reshape(1, 3)
      best_pattern_idx = pattern_order_list[np.argmin(np.sum(np.abs(error_list), axis = 1))]
      out[y, x] = best_pattern_idx
      best_error = error_list[best_pattern_idx]
      diffused_error = kernel.reshape(h_krnl, w_krnl, 3) * best_error.reshape(1, 1, 3)

      ib_y, ib_x = max(0, y - c_h_krnl), max(0, x - c_w_krnl)
      ie_y, ie_x = min(h, y + h_krnl - c_h_krnl), min(w, x + w_krnl - c_w_krnl)
      kb_y, kb_x = ib_y - (y - c_h_krnl), ib_x - (x - c_w_krnl)
      ke_y, ke_x = kb_y + ie_y - ib_y, kb_x + ie_x - ib_x
      img[ib_y:ie_y, ib_x:ie_x] -= diffused_error[kb_y:ke_y, kb_x:ke_x]
  return out


@nb.njit(cache=True, nogil=True, fastmath=True)
def convert_color(color, kernel, c_h_krnl, c_w_krnl, pattern_list, blk_size):
  dtype = color.dtype
  c, = color.shape
  _, c_pattern = pattern_list.shape
  assert c == c_pattern

  blk = np.empty((blk_size, blk_size, c), dtype = dtype)
  blk[:] = color
  converted_color = do_error_diffusion_pattern_mse(blk, kernel, c_h_krnl, c_w_krnl, pattern_list)

  out = np.empty(3, dtype = dtype)
  for i_c in range(c):
    out[i_c] = np.mean(converted_color.T[i_c])
  return out

@nb.njit(parallel=True, fastmath=True)
def convert_many_color_inplace_py(io_color_list, kernel, c_h_krnl, c_w_krnl, pattern_list, blk_size):
  #dtype = io_color_list.dtype
  n, _ = io_color_list.shape

  for i_color in nb.prange(n):
    if i_color % 100 == 0:
      print(i_color)
    io_color_list[i_color] = convert_color(io_color_list[i_color], kernel, c_h_krnl, c_w_krnl, pattern_list, blk_size)

def convert_many_color_inplace(color_list, kernel, c_h_krnl, c_w_krnl, pattern_list, blk_size, randomness):
  n_color, c = color_list.shape
  h_krnl, w_krnl, c_krnl = kernel.shape
  n_pattern, c_pattern = pattern_list.shape
  assert blk_size > 0
  assert h_krnl > 0 and w_krnl > 0
  assert h_krnl > c_h_krnl >= 0 and w_krnl > c_w_krnl >= 0
  assert c_pattern == c
  assert c_krnl == c
  assert n_pattern > 0
  if n_color <= 0:
    return
  assert randomness >= 0.0
  
  io_color_list = np.require(color_list.copy(), dtype = np.float32, requirements = "C")
  if c in (3, 4):
    pattern_list = np.require(pattern_list, dtype = np.float32, requirements = "C")
    kernel = np.require(kernel, dtype = np.float32, requirements = "C")
    if c == 3:
      _convert_many_color_inplace_mt_c3(io_color_list, n_color, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, blk_size, randomness)
    elif c == 4:
      _convert_many_color_inplace_mt_c4(io_color_list, n_color, kernel, h_krnl, w_krnl, c_h_krnl, c_w_krnl, pattern_list, n_pattern, blk_size, randomness)
  else:
    if randomness != 0.0:
      print("Fallback mode doesn't support randomness", file = sys.stderr)
    convert_many_color_inplace_py(io_color_list, kernel, c_h_krnl, c_w_krnl, pattern_list, blk_size)
  return io_color_list

def conv_to_u8(img):
  h, w, c = img.shape
  img = img * 255
  res = img - np.floor(img)
  img += (res >= np.random.uniform(low = 0.125, high = 0.875, size = (h, w, c)).astype(np.float32)).astype(np.float32)
  img = np.clip(img, 0, 255).astype(np.uint8)
  return img

@nb.njit(cache=True, nogil=True, fastmath=True)
def lerp(a, b, ratio):
  return a + (b - a) * ratio

@nb.njit(cache=True, nogil=True, fastmath=True)
def lerp_3(c000, c001, c010, c011, c100, c101, c110, c111, x, y, z):
  c00 = lerp(c000, c100, x)
  c01 = lerp(c001, c101, x)
  c10 = lerp(c010, c110, x)
  c11 = lerp(c011, c111, x)
  c0 = lerp(c00, c10, y)
  c1 = lerp(c01, c11, y)
  return lerp(c0, c1, z)

def s2lin(x):
  a = 0.055
  return np.where(x <= 0.04045, x * (1.0 / 12.92), ((x + a) * (1.0 / (1 + a))) ** 2.4)

def lin2s(x):
  a = 0.055
  return np.where(x <= 0.0031308049535603713, x * 12.92, (1 + a) * x ** (1 / 2.4) - a)

def xyz2lin(a):
  x, y, z = a.T
  r = 3.2406255 * x - 1.537208 * y - 0.4986286 * z,
  g = -0.9689307 * x + 1.8757561 * y + 0.0415175 * z,
  b = 0.0557101 * x - 0.2040211 * y + 1.0569959 * z,
  return np.array((b, g, r)).T

def lin2xyz(a):
  b, g, r = a.T
  x = 0.4124 * r + 0.3576 * g + 0.18050001 * b
  y = 0.21259999 * r + 0.71519998 * g + 0.07220002 * b
  z = 0.01930002 * r + 0.11920004 * g + 0.95050005 * b
  return np.array((x, y, z)).T

D65_XYZ = lin2xyz(np.array((1, 1, 1)))
def f_xyz2lab(t):
  return np.where(t <= (6 / 29) ** 3, 841 / 108 * t + 4 / 29, t ** (1 / 3))

def xyz2lab(a, w):
  x_n, y_n, z_n = w
  x, y, z = a.T

  f_x = f_xyz2lab(x / x_n)
  f_y = f_xyz2lab(y / y_n)
  f_z = f_xyz2lab(z / z_n)
  l = 116 * f_y - 16
  a = 500 * (f_x - f_y)
  b = 200 * (f_y - f_z)
  return np.array((l, a, b)).T

def sbgr_to_yuv(v):
  b, g, r = v.T
  y = 0.299 * r + 0.587 * g + 0.114 * b
  u = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
  v = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
  return np.array((y, u, v)).T

def yuv_to_sbgr(v):
  y, u, v = v.T
  v = v - 0.5
  u = u - 0.5
  r = y - 1.21889419e-06 * u + 1.40199959e+00 * v
  g = y - 3.44135678e-01 * u - 7.14136156e-01 * v
  b = y + 1.77200007e+00 * u + 4.06298063e-07 * v
  return np.array((b, g, r)).T

@nb.njit(cache=True, nogil=True, fastmath=True)
def sbgr_to_hsv(a):
  c = a.shape[-1]
  assert c == 3

  n = 1
  for x in a.shape[:-1]:
    n *= x

  out = a.reshape(n, c).copy()
  for i in range(n):
    b, g, r = out[i]
    v_max = np.max(out[i])
    v_min = np.min(out[i])
    if v_max == v_min:
      h = 0
    elif v_max == r:
      t = (g - b) / (v_max - v_min) / 6
      h = t if g >= b else t + 1
    elif v_max == g:
      h = (b - r) / (v_max - v_min) / 6 + 1 / 3
    elif v_max == b:
      h = (r - g) / (v_max - v_min) / 6 + 2 / 3
    s = 0 if v_max == 0 else 1 - v_min / v_max
    v = v_max
    out[i, 0] = h
    out[i, 1] = s
    out[i, 2] = v
  return out.reshape(*a.shape)

def hsv_to_sbgr(a):
  h, s, v = a.T

  h6 = h * 6
  h_i = np.floor(h6) % 6
  f = h6 - h_i
  p = v * (1 - s)
  q = v * (1 - f * s)
  t = v * (1 - (1 - f) * s)

  return np.where(
    h_i == 0, np.array((p, t, v)),
    np.where(h_i == 1, np.array((p, v, q)),
      np.where(h_i == 2, np.array((t, v, p)),
        np.where(h_i == 3, np.array((v, q, p)),
          np.where(h_i == 4, np.array((v, p, t)),
            np.array((q, p, v))
          )
        )
      )
    )
  ).T