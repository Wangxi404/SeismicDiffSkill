#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "$(uname -s)" in
  Darwin) libflag=-dynamiclib; ext=dylib;;
  *) libflag=-shared; ext=so;;
esac
"${CXX:-clang++}" -O2 -std=c++17 -fPIC "$libflag" native_wave.cpp -o "libnative.$ext"
"${CXX:-clang++}" -O2 -std=c++17 -fPIC "$libflag" native_wave.cpp adjoint.cpp -o "libadjoint.$ext"
