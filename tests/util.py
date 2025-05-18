import pytest
from time import perf_counter

import torch


def assert_close(x, y, tol, name: str):
    assert x.shape == y.shape, (x.shape, y.shape)
    err = torch.amax(torch.abs(y - x)).item()
    print(f"{name}: maxerr = {err:.2g}")
    assert err <= tol


def timeit(fun, name: str, repeat=20):
    torch.cuda.empty_cache()

    for i in range(2):
        fun()
    torch.cuda.synchronize()

    time0 = perf_counter()
    for i in range(repeat):
        fun()
        torch.cuda.synchronize()
    time1 = perf_counter()

    dt = 1e3 * (time1-time0) / repeat
    print(f"{name}: {dt:.2f} ms")
