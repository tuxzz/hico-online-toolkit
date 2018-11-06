import numpy as np
import pickle

path = "cube_8x16x16x16_rgb_rgb_q256.pickle"

cube = pickle.load(open(path, "rb"))
out = (cube * 1024).astype(np.float16).tobytes()
open("out.bin", "wb").write(out)