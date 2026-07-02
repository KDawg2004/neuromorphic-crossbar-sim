import numpy as np

def make_vertical(col):
    img = np.zeros((4, 4))
    img[:, col] = 1
    return img.flatten()

def make_horizontal(row):
    img = np.zeros((4, 4))
    img[row, :] = 1
    return img.flatten()

def make_diagonal():
    return np.eye(4).flatten()

def make_x():
    img = np.eye(4) + np.fliplr(np.eye(4))
    img = (img > 0).astype(float)
    return img.flatten()

import numpy as np

rng = np.random.default_rng(0)

def add_noise(img, p=0.1):
    flip = rng.random(img.shape) < p
    return np.where(flip, 1 - img, img)

X = []
y = []

for col in range(4):
    for _ in range(3):
        X.append(add_noise(make_vertical(col))); y.append(0)
for row in range(4):
    for _ in range(3):
        X.append(add_noise(make_horizontal(row))); y.append(1)
for _ in range(6):
    X.append(add_noise(make_diagonal())); y.append(2)
for _ in range(6):
    X.append(add_noise(make_x())); y.append(3)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)