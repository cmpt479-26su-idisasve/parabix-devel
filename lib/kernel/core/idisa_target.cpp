/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>

#include <toolchain/toolchain.h>
#include <idisa/idisa_i64_builder.h>
#ifdef PARABIX_ARM_TARGET
#include <idisa/idisa_arm_builder.h>
#include <idisa/idisa_sve_builder.h>
#endif
#ifdef PARABIX_X86_TARGET
#include <idisa/idisa_sse_builder.h>
#include <idisa/idisa_avx_builder.h>
#endif
#ifdef PARABIX_NVPTX_TARGET
#include <idisa/idisa_nvptx_builder.h>
#endif
#include <llvm/IR/Module.h>

#include <llvm/TargetParser/Triple.h>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>

#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif

using namespace kernel;
using namespace llvm;

namespace IDISA {

KernelBuilder * GetIDISA_Builder(llvm::LLVMContext & C, const codegen::FeatureSet & featureSet) {
    if (((codegen::BlockSize & (codegen::BlockSize - 1)) != 0) || (codegen::BlockSize < 64)) {
        llvm::report_fatal_error("BlockSize must be a power of 2 and >=64");
    }

    if (codegen::BlockSize == 64) {
        return new KernelBuilderImpl<IDISA_I64_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

#if defined(PARABIX_X86_TARGET)
    // AVX512BW builder can only be used for BlockSize multiples of 512
    if (codegen::BlockSize >= 512 && featureSet.test((size_t)codegen::Feature::AVX512F)) {
        return new KernelBuilderImpl<IDISA_AVX512F_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
    if (codegen::BlockSize >= 256) {
        // AVX2 or AVX builders can only be used for BlockSize multiples of 256
        if (featureSet.test((size_t)codegen::Feature::AVX2)) {
            return new KernelBuilderImpl<IDISA_AVX2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        } else if (featureSet.test((size_t)codegen::Feature::AVX)) {
            return new KernelBuilderImpl<IDISA_AVX_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        }
    }
    if (codegen::BlockSize == 128) {
        if (featureSet.test((size_t)codegen::Feature::SSSE3)) {
            return new KernelBuilderImpl<IDISA_SSSE3_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        } else {
            return new KernelBuilderImpl<IDISA_SSE2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        }
    }
    // Otherwise, fall through...
    llvm::errs() << "BlockSize 64 default!\n";
    codegen::BlockSize = 64;
    return new KernelBuilderImpl<IDISA_I64_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
#elif defined(PARABIX_ARM_TARGET)
    // Try for SVE/SVE2
    if (featureSet.test((size_t)codegen::Feature::SVE)) {
        return new KernelBuilderImpl<IDISA_SVE_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
    // As of July 2026, aarch64 is supposed to always include Neon
    return new KernelBuilderImpl<IDISA_ARM_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
#endif
}

#ifdef PARABIX_NVPTX_TARGET
KernelBuilder * GetIDISA_GPU_Builder(llvm::LLVMContext & C) {
    return new KernelBuilderImpl<IDISA_NVPTX20_Builder>(C, 64 * 64, 64);
}
#endif

}
