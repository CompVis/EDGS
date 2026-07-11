#!/usr/bin/env python
"""Minimal test to verify EDGS environment"""

print("Testing EDGS environment...")

# Test basic imports
try:
    import os
    import sys
    print("✓ Basic imports OK")
except Exception as e:
    print(f"✗ Basic import error: {e}")

# Test numpy
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy error: {e}")

# Test torch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

# Test hydra
try:
    import hydra
    print(f"✓ Hydra imported")
except Exception as e:
    print(f"✗ Hydra error: {e}")

# Test moviepy
try:
    import moviepy
    print(f"✓ MoviePy imported")
except Exception as e:
    print(f"✗ MoviePy error: {e}")

# Test pycolmap
try:
    import pycolmap
    print(f"✓ PyColmap imported")
except Exception as e:
    print(f"✗ PyColmap error: {e}")

print("\nEnvironment test complete!")