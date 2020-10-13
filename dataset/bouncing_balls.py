"""
Taken FROM @mbeissinger:
https://github.com/mbeissinger/recurrent_gsn/blob/master/src/data/bouncing_balls.py

This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""
from math import sqrt

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from numpy import shape, array, zeros, dot, arange, meshgrid, exp
from numpy.random import randn, rand

SIZE = 10  # size of bounding box: SIZE X SIZE.


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x):
    return sqrt((x ** 2).sum())


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def bounce_n(steps=128, n=2, r=None, m=None):
    if r is None:
        r = array([1.2] * n).astype("float32")
    if m is None:
        m = array([1] * n).astype("float32")
    # r is to be rather small.
    x = zeros((steps, n, 2), dtype="float32")
    v = randn(n, 2).astype("float32")
    v = v / norm(v) * 0.5
    good_config = False
    while not good_config:
        _x = 2 + rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if _x[i][z] - r[i] < 0:
                    good_config = False
                if _x[i][z] + r[i] > SIZE:
                    good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(_x[i] - _x[j]) < r[i] + r[j]:
                    good_config = False

    eps = 0.5
    for t in range(steps):
        # for how long do we show small simulation
        for i in range(n):
            x[t, i] = _x[i]

        for mu in range(int(1 / eps)):
            for i in range(n):
                _x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if _x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if _x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(_x[i] - _x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = _x[i] - _x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)
    return x.astype("float32")


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype="float32")


def matricize(x, res, r=None):
    steps, n = shape(x)[0:2]
    if r is None:
        r = array([1.2] * n).astype("float32")

    a = zeros((steps, res, res), dtype="float32")

    [i, j] = meshgrid(ar(0, 1, 1.0 / res) * SIZE, ar(0, 1, 1.0 / res) * SIZE)

    for t in range(steps):
        for ball in range(n):
            ball_x, ball_y = x[t, ball]
            _delta = exp(
                -((((i - ball_x) ** 2 + (j - ball_y) ** 2) / (r[ball] ** 2)) ** 4)
            )
            a[t] += _delta

        a[t][a[t] > 1] = 1
    return a


def bounce_mat(res, n=2, steps=128, r=None):
    if r is None:
        r = array([1.2] * n).astype("float32")
    x = bounce_n(steps, n, r)
    a = matricize(x, res, r)
    return a


def bounce_vec(res, n=2, steps=128, r=None, m=None):
    if r is None:
        r = array([1.2] * n).astype("float32")
    x = bounce_n(steps, n, r, m)
    v = matricize(x, res, r)
    v = v.reshape(steps, res ** 2)
    return v


def sample_sequence_mat(sequence, step=5, output_frames=6):
    startIdx = sequence.shape[0] // 3
    stopIdx = startIdx + step * output_frames + 1
    # if stopIdx < sequence.shape[0]:
    new_sequence = sequence[startIdx:stopIdx:step]
    new_sequence = new_sequence.unsqueeze(dim=1)
    new_sequence = new_sequence.repeat(1, 3, 1, 1)
    return new_sequence


class BouncingBalls(data.Dataset):
    def __init__(
        self,
        size=15,
        timesteps=128,
        n_balls=3,
        flatten=False,
        mode="train",
        train_size=2000,
        transform=None,
        output_frames=6,
    ):
        super().__init__()
        self.size = size
        self.timesteps = timesteps
        self.n_balls = n_balls
        self.flatten = flatten
        self.mode = mode
        self.train_size = train_size
        self.transform = transform
        self.output_frames = output_frames

    def __getitem__(self, index):
        np.random.seed(hash(self.mode) % 2 ** 10 + index)
        if self.flatten:
            sample = bounce_vec(res=self.size, n=self.n_balls, steps=self.timesteps)
        else:
            sample = bounce_mat(res=self.size, n=self.n_balls, steps=self.timesteps)
            sample = torch.from_numpy(sample)
            sample = sample_sequence_mat(sample, output_frames=self.output_frames)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # arbitrary since we make a new sequence each time
        return (
            int(self.train_size) if self.mode == "train" else int(self.train_size * 0.2)
        )


if __name__ == "__main__":
    bounce_data = BouncingBalls(224, 128, 3)
    images = [Image.fromarray(step * 255) for step in bounce_data[0]]
    im = images[0]
    rest = images[1::2]
    with open("./test_224_128_3_x2.gif", "wb") as f:
        im.save(f, save_all=True, append_images=rest)
