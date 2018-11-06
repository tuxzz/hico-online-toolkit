import numpy as np
import cv2
import ediff

img_path = "a.png"
img_path = "GM0OszQM.jpg"
p_low, p_high = -0.1, 0.1
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
img = cv2.imread(img_path).astype(np_dtype) / 255
h, w, c = img.shape
assert c == 3

print("Generate noise_tex...")
#t_filter = sp.firwin(9, 0.5 - 1e-5, window = "nuttall", pass_zero = False, scale = True, nyq = 1.0)
noise_tex = np.random.uniform(low = p_low, high = p_high, size = (h, w, c)).astype(np_dtype)
#p_tex = conv_2d(p_tex, t_filter)

print("Add noise...")
#img += noise_tex

print("Begin...")
kernel = kernel.repeat(3).reshape(*kernel.shape, 3)
img = ediff.do_error_diffusion_pattern_mse(img, kernel, c_h_krnl, c_w_krnl, pattern_list)
cv2.imwrite("out.png", img * 255)
print("Done!")