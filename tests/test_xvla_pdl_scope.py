"""Compile the actual scope header: nesting, exceptions, threads and multiple TUs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_scope_restores_state_and_is_thread_local(tmp_path):
    compiler = shutil.which("g++")
    nvcc = shutil.which("nvcc")
    cuda = os.environ.get("CUDA_HOME")
    if not compiler or not (cuda or nvcc):
        pytest.skip("C++ compiler and CUDA headers required")
    include = Path(cuda) / "include" if cuda else Path(nvcc).resolve().parents[1] / "include"
    root = Path(__file__).resolve().parents[1]
    other = tmp_path / "other.cpp"
    other.write_text('#include "pdl_switch.cuh"\nbool other_flag() { return flash_rt::pdl::flag(); }\n')
    source = tmp_path / "scope.cpp"
    source.write_text(r'''#include "pdl_switch.cuh"
#include <cassert>
#include <thread>
#include <stdexcept>
bool other_flag();
int main() {
  using flash_rt::pdl::flag;
  using flash_rt::pdl::Scope;
  assert(!flag() && !other_flag());
  {
    Scope outer(true); outer.enter();
    assert(flag() && other_flag());
    { Scope nested(false); nested.enter(); assert(!flag()); }
    assert(flag());
    std::thread worker([] { assert(!flag()); Scope s(true); s.enter(); assert(flag()); });
    worker.join();
    assert(flag());
    try { Scope nested(false); nested.enter(); throw std::runtime_error("failure"); }
    catch (const std::runtime_error&) { assert(flag()); }
    outer.leave(); outer.leave(); assert(!flag());
  }
  assert(!flag() && !other_flag());
}
''')
    binary = tmp_path / "scope"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I"+str(include),
                    "-I"+str(root/"csrc/kernels"), str(source), str(other), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
