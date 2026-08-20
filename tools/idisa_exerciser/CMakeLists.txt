#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: idisa_exerciser

parabix_add_executable(
NAME
    idisa_exerciser
SRC
    idisa_exerciser.cpp
DEPS
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.util
)
