import numpy as np
import numba as nb
import cv2
import ediff
import pickle

img_path = "GM0OszQM_hd.png"
#img_path = "lolita.jpg"
np_dtype = np.float32
diffuse_fac = 1

print("Load LUT Cube...")
with open("out_cube.pickle", "rb") as f:
  cube = pickle.load(f).astype(np_dtype)
n_diffuse_fac, n_s, n_p, n_t, c_cube = cube.shape

print("Load Image...")
img = cv2.imread(img_path).astype(np_dtype) / 255
h, w, c = img.shape
assert c == c_cube

print("Convert...")
i_diffuse_fac = diffuse_fac * (n_diffuse_fac - 1)
f_diffuse_fac = int(i_diffuse_fac)
c_diffuse_fac = int(np.ceil(i_diffuse_fac))
r_diffuse_fac = i_diffuse_fac - f_diffuse_fac

@nb.njit(parallel=True, fastmath=True)
def do(img):
  max_s, max_p, max_t = n_s - 1, n_p - 1, n_t - 1
  f_img = img.reshape(h * w, c)
  for i_pix in nb.prange(h * w):
      vs, vp, vt = f_img[i_pix]
      s, p, t = vs * max_s, vp * max_p, vt * max_t
      s, p, t = max(0.0, min(s, max_s)), max(0.0, min(p, max_p)), max(0.0, min(t, max_t))
      f_s, f_p, f_t = int(s), int(p), int(t)
      r_s, r_p, r_t = s - f_s, p - f_p, t - f_t
      c_s, c_p, c_t = int(np.ceil(s)), int(np.ceil(p)), int(np.ceil(t))
      
      cube_l = cube[f_diffuse_fac]
      cube_h = cube[c_diffuse_fac]
      l_diff = ediff.lerp_3(
        cube_l[f_s, f_p, f_t], cube_l[f_s, f_p, c_t], cube_l[f_s, c_p, f_t], cube_l[f_s, c_p, c_t],
        cube_l[c_s, f_p, f_t], cube_l[c_s, f_p, c_t], cube_l[c_s, c_p, f_t], cube_l[c_s, c_p, c_t],
        r_s, r_p, r_t,
      )
      h_diff = ediff.lerp_3(
        cube_h[f_s, f_p, f_t], cube_h[f_s, f_p, c_t], cube_h[f_s, c_p, f_t], cube_h[f_s, c_p, c_t],
        cube_h[c_s, f_p, f_t], cube_h[c_s, f_p, c_t], cube_h[c_s, c_p, f_t], cube_h[c_s, c_p, c_t],
        r_s, r_p, r_t,
      )
      f_img[i_pix] = ediff.lerp(l_diff, h_diff, r_diffuse_fac)

#img = ediff.sbgr_to_yuv(img).copy()
do(img)
#img = ediff.yuv_to_sbgr(img).copy()
cv2.imwrite("out.png", ediff.conv_to_u8(img))
print("Done")