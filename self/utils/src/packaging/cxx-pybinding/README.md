# fastla

C++ linear algebra kernels (`dot`, `norm2`, `axpy`) exposed to Python with pybind11.

Binding engine: pybind11.

## Quickstart

Requirements: a C++17 compiler, CMake >= 3.18, Python >= 3.8.

There are two ways to build and load the package. Option A is the
default for development.

### Option A - CMake build + PYTHONPATH (no install)

```bash
# 1. Install the dependencies
pip install pybind11 numpy pytest

# 2. Configure and build the Python module
cd fastla
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build -j

# 3. Put the compiled module inside the Python package
cp build/_core*.so src/fastla/

# 4. Run, telling Python to look for packages in src/
PYTHONPATH=src python examples/demo.py
PYTHONPATH=src python -m pytest
```

`PYTHONPATH=src` adds the `src/` folder to Python's module search path for
that one command, so `import fastla` finds `src/fastla/`. To keep it for the
whole terminal session instead:

```bash
export PYTHONPATH=$PWD/src
python examples/demo.py
```

After editing any C++ file, repeat steps 2 and 3 (`cmake --build build -j`
and the `cp`).

### Option B - install as a Python package (pip)

```bash
# 1. Install the build tools
pip install scikit-build-core pybind11 numpy pytest

# 2. Build and install fastla (editable install)
cd fastla
pip install -e . --no-build-isolation

# 3. Run (no PYTHONPATH needed: fastla is now installed)
python examples/demo.py
pytest
```

After editing any C++ file, run `pip install -e . --no-build-isolation` again.

Note: do not mix the two options. If fastla is installed with pip, uninstall
it (`pip uninstall fastla`) before using Option A, so Python does not load
the installed copy.

### Use it from Python

```python
import numpy as np
import fastla

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

fastla.dot(x, y)        # 32.0
fastla.norm2(x)         # 3.7416...
fastla.axpy(2.0, x, y)  # y is now [6, 9, 12] (modified in place)
```

## Layout

```
fastla/
+-- CMakeLists.txt            one build: C++ core, Python module, C++ tests
+-- pyproject.toml            pip build via scikit-build-core
+-- cpp/
|   +-- include/fastla/       public C++ headers, used as #include <fastla/...>
|   +-- src/                  C++ implementation (no Python dependency)
|   +-- tests/                C++ unit tests (Catch2)
+-- bindings/module.cpp       pybind11 glue, compiled into fastla._core
+-- src/fastla/               Python package
|   +-- __init__.py           re-exports the C++ functions
|   +-- _core.pyi             type stubs for editors / mypy
|   +-- py.typed
+-- tests/                    pytest tests for the bindings
+-- benchmarks/               fastla vs NumPy
+-- examples/demo.py
```

## Benchmarks

```bash
PYTHONPATH=src python benchmarks/bench_dot.py   # Option A
python benchmarks/bench_dot.py                  # Option B
```

## C++ only: build and test (no Python needed)

```bash
cmake -S . -B build -DFASTLA_BUILD_PYTHON=OFF -DFASTLA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Catch2 is downloaded automatically by CMake (FetchContent), so this step
needs internet access the first time.

## CMake options

| Option                | Default | Meaning                                    |
|-----------------------|---------|--------------------------------------------|
| `FASTLA_BUILD_PYTHON` | ON      | Build the Python extension module          |
| `FASTLA_BUILD_TESTS`  | OFF     | Build the C++ unit tests                   |
| `FASTLA_NATIVE`       | OFF     | `-march=native` (fast, but not portable)   |

Enable native tuning with:

```bash
# Option A
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DFASTLA_NATIVE=ON
# Option B
pip install -e . --no-build-isolation -Ccmake.define.FASTLA_NATIVE=ON
```

## Adding a new function

1. Declare it in `cpp/include/fastla/`, implement it in `cpp/src/`.
2. Add a C++ test in `cpp/tests/`.
3. Bind it with `m.def(...)` in `bindings/module.cpp`.
4. Export it in `src/fastla/__init__.py` and add its signature to `_core.pyi`.
5. Add a pytest test in `tests/`.
