# TorchScript-TensorRT-SSD
An example on how to run SSD for inference with TensorRT and TorchScript

This notebook demonstrates how to use PyTorch and TRTorch (https://www.github.com/NVIDIA/TRTorch) to optimize 
an SSD model and still run it with PyTorch APIs. 

### Dependencies

- PyTorch: `1.5.0+`
- TRTorch: > `0.0.2` (may need to compile from source if there is no release after `0.0.2` yet, instructions are in the repo)
- TensorRT `7.0`

### Benchmarking SSD

You can compare the performance of the compiled TRT module with a normal TorchScript module using the benchmark application included
in the TRTorch repository. After running this notebook you will see that a file has been created with the serialized
TorchScript code and weights called `ssd_300_traced.jit.pt`. This can be run in the benchmark application like so:

```sh
# From within the TRTorch repo directory
bazel run //cpp/benchmark --compilation_mode=opt --cxxopt="-DTRT" --cxxopt="-DJIT" -- $(realpath <PATH TO JIT FILE>/ssd_300_traced.jit.pt) "(32 3 300 300)"
```

This runs in batch size 32, for a `3x300x300` input, this can be changed by changing the desired dimension in the last argument by 
the input size must be greater than `3x300x300`
