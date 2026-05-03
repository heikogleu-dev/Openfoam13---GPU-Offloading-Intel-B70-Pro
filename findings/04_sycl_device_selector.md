# Bug: Wrong ONEAPI_DEVICE_SELECTOR Syntax Causes Crash

## Symptom

```
sycl::_V1::exception: Error parsing selector string "level_zero:gpu:0"
                       Too many colons (:)
```

All MPI ranks crash immediately at the `Foam::Time` constructor (very early,
before mesh loaded).

## Wrong Syntax

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu:0   # CRASHES — 3 colons
```

The Intel docs example syntax `<backend>:<device_type>` doesn't combine
with explicit device index — only one of the two filter forms works at a time.

## Correct Syntaxes

```bash
# Pick by device index (preferred for compute):
export ONEAPI_DEVICE_SELECTOR=level_zero:0       # device 0 of L0 backend

# Or pick by type (less specific):
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu     # all GPUs of L0 backend
```

## Context

With iGPU active, `sycl-ls` shows:
```
[level_zero:gpu][level_zero:0] Intel(R) Graphics [0xe223]    ← B70 Pro
[level_zero:gpu][level_zero:1] Intel(R) Graphics              ← Arrow Lake iGPU
[opencl:cpu]    [opencl:0]     Intel(R) Core(TM) Ultra 9 285K
[opencl:gpu]    [opencl:1]     Intel(R) Graphics [0xe223]    ← B70 Pro again
[opencl:gpu]    [opencl:2]     Intel(R) Graphics              ← iGPU again
```

Without explicit selector, OGL's `compute_device_id()` does round-robin
across all devices SYCL reports — causing 4 of 8 ranks to land on the iGPU
(slow!) and producing ~15× total performance loss.

**Always specify `level_zero:0` for the B70 Pro.**

## Verification

After setting `ONEAPI_DEVICE_SELECTOR=level_zero:0`, sycl-ls shows only
the B70 Pro:
```
[level_zero:gpu][level_zero:0] Intel(R) Graphics [0xe223]
```
