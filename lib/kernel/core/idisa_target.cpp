/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>

#include <idisa/idisa_i64_builder.h>
#include <toolchain/toolchain.h>
#ifdef PARABIX_ARM_TARGET
#include <idisa/idisa_neon_builder.h>
#include <idisa/idisa_sve_builder.h>
#include <llvm/TargetParser/AArch64TargetParser.h>
#endif
#ifdef PARABIX_X86_TARGET
#include <idisa/idisa_avx_builder.h>
#include <idisa/idisa_sse_builder.h>
#endif
#ifdef PARABIX_NVPTX_TARGET
#include <idisa/idisa_nvptx_builder.h>
#endif
#include <llvm/IR/Module.h>

#include <llvm/TargetParser/Triple.h>

#include <kernel/core/kernel_builder.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif

using namespace kernel;
using namespace llvm;

namespace IDISA {

KernelBuilder *GetIDISA_Builder(llvm::LLVMContext &C, const StringMap<bool> &features) {
    if (((codegen::BlockSize & (codegen::BlockSize - 1)) != 0) || (codegen::BlockSize < 64)) {
        llvm::report_fatal_error("BlockSize must be a power of 2 and >=64");
    }

    codegen::FeatureSet featureSet = codegen::MapFeatureNames(features);

    if ((codegen::BlockSize == 64) || codegen::UseI64Builder) {
        return new KernelBuilderImpl<IDISA_I64_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

#ifdef PARABIX_X86_TARGET
    // AVX512BW builder can only be used for BlockSize multiples of 512
    if (featureSet.test((size_t)codegen::Feature::AVX512F) && (codegen::BlockSize >= 512)) {
        return new KernelBuilderImpl<IDISA_AVX512F_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

    // AVX2 or AVX builders can only be used for BlockSize multiples of 256
    if (featureSet.test((size_t)codegen::Feature::AVX2) && (codegen::BlockSize >= 256)) {
        return new KernelBuilderImpl<IDISA_AVX2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
    if (featureSet.test((size_t)codegen::Feature::AVX) && (codegen::BlockSize >= 256)) {
        return new KernelBuilderImpl<IDISA_AVX_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

    if (featureSet.test((size_t)codegen::Feature::SSSE3) && (codegen::BlockSize >= 128)) {
        return new KernelBuilderImpl<IDISA_SSSE3_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

    // We excluded <128 earlier
    assert(codegen::BlockSize >= 128);

    // SSE2 always available on x86-64
    return new KernelBuilderImpl<IDISA_SSE2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
#elif defined(PARABIX_ARM_TARGET)
    // Try for SVE/SVE2
    if (featureSet.test((size_t)codegen::Feature::SVE)) {
        return new KernelBuilderImpl<IDISA_SVE_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

    // We excluded <128 earlier
    assert(codegen::BlockSize >= 128);

    // As of July 2026, aarch64 is supposed to always include Neon
    return new KernelBuilderImpl<IDISA_Neon_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
#else
#error Unknown target type?
#endif
}

#if defined(PARABIX_NVPTX_TARGET)
KernelBuilder *GetIDISA_GPU_Builder(llvm::LLVMContext &C) {
    return new KernelBuilderImpl<IDISA_NVPTX20_Builder>(C, featureSet, 64 * 64, 64);
}
#endif

} // namespace IDISA
