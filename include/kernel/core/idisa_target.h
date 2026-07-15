/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <llvm/Support/Compiler.h>
#include <llvm/ADT/StringMap.h>
#include <toolchain/toolchain.h>

namespace llvm { class LLVMContext; }
namespace kernel { class KernelBuilder; }

namespace IDISA {

kernel::KernelBuilder * GetIDISA_Builder(llvm::LLVMContext & C, const codegen::FeatureSet & featureSet);

#ifdef CUDA_ENABLED
kernel::KernelBuilder * GetIDISA_GPU_Builder(llvm::LLVMContext & C);
#endif

}

