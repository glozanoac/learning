# fastla

C++ kernels (`dot`, `norm2`, `axpy`) exposed to Python with pybind11.

```
fastla/
├── CMakeLists.txt              # builds C++ lib + Python module
├── pyproject.toml              # pip build via scikit-build-core
├── src/
│   ├── fastla_core/            # pure C++ (no Python)
│   │   ├── vector_ops.hpp
│   │   └── vector_ops.cpp
│   └── bindings.cpp            # pybind11 glue -> module fastla._core
├── python/fastla/__init__.py   # Python package re-exporting the C++ functions
└── examples/demo.py
```

## Option A — install as a Python package (recommended)

```bash
pip install .          # or: pip install -e . --no-build-isolation
python examples/demo.py
```

## Option B — plain CMake build

```bash
pip install pybind11 numpy
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build -j
cp build/_core*.so python/fastla/
PYTHONPATH=python python examples/demo.py
```

## Adding a new function

1. Declare/implement it in `src/fastla_core/`.
2. Add an `m.def(...)` in `src/bindings.cpp`.
3. Export it in `python/fastla/__init__.py`.
