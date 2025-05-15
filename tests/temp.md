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
