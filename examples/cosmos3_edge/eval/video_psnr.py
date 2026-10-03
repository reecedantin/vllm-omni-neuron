"""PSNR between two videos: python psnr.py ref.mp4 test.mp4 [sheet.png]"""
import sys

import cv2
import numpy as np


def rd(p):
    c = cv2.VideoCapture(p)
    f = []
    while True:
        ok, x = c.read()
        if not ok:
            break
        f.append(x.astype(np.float32))
    return np.stack(f)


a, b = rd(sys.argv[1]), rd(sys.argv[2])
n = min(len(a), len(b))
per = [10 * np.log10(255 ** 2 / max(((a[i] - b[i]) ** 2).mean(), 1e-6)) for i in range(n)]
m = ((a[:n] - b[:n]) ** 2).mean()
print(f"{sys.argv[2]}: frames {n} PSNR {10 * np.log10(255 ** 2 / m):.2f} dB "
      f"(min frame {min(per):.2f}, first {per[0]:.2f}, last {per[-1]:.2f})")
if len(sys.argv) > 3:
    idx = [0, n // 2, n - 1]
    sheet = np.concatenate([np.concatenate([a[i], b[i]], 1) for i in idx], 0).astype(np.uint8)
    cv2.imwrite(sys.argv[3], sheet)
