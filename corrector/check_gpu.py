"""Standalone GPU sanity check.

Run:
    python -m corrector.check_gpu

Reports:
  - PyTorch version, CUDA build version
  - cuda.is_available()
  - device count, name, capability
  - a quick allocate + matmul test on the GPU
"""
import sys
import time


def main():
    try:
        import torch
    except ImportError as e:
        print("FAIL: torch is not installed in this environment.")
        print(f"  ({e})")
        sys.exit(2)

    print(f"torch version : {torch.__version__}")
    print(f"compiled CUDA : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA-capable GPU visible to PyTorch.")
        print("Training will run on CPU (much slower for the MLP).")
        sys.exit(1)

    n = torch.cuda.device_count()
    print(f"device count  : {n}")
    for i in range(n):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}  "
              f"cap={torch.cuda.get_device_capability(i)}")

    # Quick smoke test: allocate, matmul, sync
    print("\nrunning matmul smoke test on cuda:0 ...")
    dev = torch.device("cuda:0")
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  2048x2048 @ 2048x2048 in {dt*1000:.1f} ms  "
          f"(~{2*2048**3/dt/1e9:.0f} GFLOP/s)")
    print("  result tensor shape:", tuple(c.shape), " dtype:", c.dtype)
    print("\nGPU OK.")


if __name__ == "__main__":
    main()
