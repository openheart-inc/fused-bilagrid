May 14 initial commit:

```md
# Test grid sample
output: maxerr = 7.6e-06
input.grad: maxerr = 2.9e-10
grid.grad: maxerr = 1.2e-10
rgb.grad: maxerr = 5.4e-06

# Profile forward
forward torch: 4.40 ms
forward fused: 0.52 ms

# Profile backward
backward torch: 18.59 ms
backward fused: 12.05 ms
```

May 15 commit:

```md
# Test grid sample
output: maxerr = 4.8e-06
input.grad: maxerr = 2.2e-10
grid.grad: maxerr = 1.6e-10
rgb.grad: maxerr = 3.2e-12

# Profile forward
forward torch: 4.60 ms
forward fused: 0.45 ms

# Profile backward
backward torch: 18.35 ms
backward fused: 8.11 ms
```

May 16 commit:

```md
# Test sample forward
output: maxerr = 4.4e-06

# Test sample backward
bilagrid.grad: maxerr = 2.3e-10
rgb.grad: maxerr = 8.4e-11

# Test sample backward, with coords.grad
bilagrid.grad: maxerr = 1.8e-10
coords.grad: maxerr = 1.3e-10
rgb.grad: maxerr = 8.4e-11

# Test uniform sample forward
output: maxerr = 1.1e-05

# Test uniform sample backward
bilagrid.grad: maxerr = 3.9e-11
rgb.grad: maxerr = 3.6e-11


# Profiling sample

torch forward: 4.99 ms
fused forward: 0.44 ms

torch backward: 18.64 ms
fused backward: 7.20 ms
fused backward with coords.grad: 7.33 ms

# Profiling uniform sample

torch forward: 5.02 ms
fused forward: 0.42 ms

torch backward: 20.02 ms
fused backward: 31.99 ms
```
