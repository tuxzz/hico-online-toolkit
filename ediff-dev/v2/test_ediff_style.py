import numpy as np
import cv2
import ediff

img_path = "GM0OszQM_hd.png"
randomness = 0.1
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

# choose rgb in yuv pattern
'''pattern_list = np.array([
  [1, 0, 0],
  [1, 1, 0],
  [1, 0, 1],
  [1, 1, 1],
  [0.   , 0.5  , 0.5  ],
  [1.   , 0.5  , 0.5  ]
], dtype = np_dtype)'''
pattern_list = ediff.sbgr_to_yuv(pattern_list)

kernel *= 0.25
print("Load Image...")
img = cv2.imread(img_path).astype(np_dtype) / 255
h, w, c = img.shape
assert c == 3

print("Begin... %d" % (h * w))
kernel = kernel.repeat(c).reshape(*kernel.shape, c)
img = ediff.sbgr_to_yuv(img).copy()
out = ediff.convert_many_color_inplace(img.reshape(h * w, c), kernel, c_h_krnl, c_w_krnl, pattern_list, 32, randomness).reshape(h, w, c)
out = ediff.yuv_to_sbgr(out).copy()
cv2.imwrite("out.png", ediff.conv_to_u8(out))
print("Done!")