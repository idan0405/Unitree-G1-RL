"""Debug helper: try to cloudpickle-load a file given as argv[1], with a shim
for a NumPy Generator unpickling quirk. Used to diagnose checkpoint load errors.

Usage: python tests/test_pickle.py <path-to-pickle>
"""

import cloudpickle
import sys
import numpy as np

if hasattr(np.random, "_pickle"):
    original_ctor = np.random._pickle.__generator_ctor
    def patched_ctor(*args, **kwargs):
        return original_ctor(args[0] if args else "PCG64")
    np.random._pickle.__generator_ctor = patched_ctor

def main():
    path = sys.argv[1]
    print(f"Loading {path} with cloudpickle...")
    with open(path, "rb") as f:
        obj = cloudpickle.load(f)
    print("Success!")

if __name__ == "__main__":
    main()
