# Fully Fused Differentiable Bilateral Grid

```bash
pip install . -v --no-build-isolation
```


## Consistency with PyTorch implementation

Bilateral grid uses bilinear interpolation, which is C0 continuous and therefore involves undefined gradient, especially with respect to sample coordinates. Fused-bilagrid detects such discontinuity and sets such gradient to zero, while PyTorch typically assigns an arbitrary value. This does not affect Gaussian splatting training since gradient of sample coordinates is unused.
