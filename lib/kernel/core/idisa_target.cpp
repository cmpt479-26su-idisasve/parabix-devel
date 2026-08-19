/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>

#include <toolchain/toolchain.h>
#include <idisa/idisa_i64_builder.h>
#ifdef PARABIX_ARM_TARGET
#include <llvm/TargetParser/AArch64TargetParser.h>
#include <idisa/idisa_neon_builder.h>
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

#define ADD_IF_FOUND(Flag, Value) if (features.lookup(Value)) featureSet.set((size_t)Feature::Flag)

using namespace kernel;
using namespace llvm;

struct Features {
    bool hasAVX;
    bool hasAVX2;
    bool hasAVX512F;

    Features() : hasAVX(0), hasAVX2(0), hasAVX512F(0) {}
};

Features getHostCPUFeatures(const StringMap<bool> & features) {
    Features hostCPUFeatures;
    hostCPUFeatures.hasAVX = features.lookup("avx");
    hostCPUFeatures.hasAVX2 = features.lookup("avx2");
    hostCPUFeatures.hasAVX512F = features.lookup("avx512f");
    return hostCPUFeatures;
}

bool AVX2_available() {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(19, 0, 0)
    StringMap<bool> features;
    if (LLVM_UNLIKELY(!sys::getHostCPUFeatures(features))) {
        return false;
    }
    #else
    const auto features = sys::getHostCPUFeatures();
    #endif
    return features.lookup("avx2");
}

bool AVX512BW_available() {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(19, 0, 0)
    StringMap<bool> features;
    if (LLVM_UNLIKELY(!sys::getHostCPUFeatures(features))) {
        return false;
    }
    #else
    const auto features = sys::getHostCPUFeatures();
    #endif
    return features.lookup("avx512bw");
}

namespace IDISA {

KernelBuilder * GetIDISA_Builder(llvm::LLVMContext & C, const StringMap<bool> & features) {
    if (((codegen::BlockSize & (codegen::BlockSize - 1)) != 0) || (codegen::BlockSize < 64)) {
        llvm::report_fatal_error("BlockSize must be a power of 2 and >=64");
    }

    codegen::FeatureSet featureSet = codegen::MapFeatureNames(features);

    if (codegen::BlockSize == 64) {
        return new KernelBuilderImpl<IDISA_I64_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }

#ifdef PARABIX_X86_TARGET
    const auto HasAVX = featureSet.test((size_t)codegen::Feature::AVX);
    const auto HasAVX2 = featureSet.test((size_t)codegen::Feature::AVX2);
    const auto HasAVX512F = featureSet.test((size_t)codegen::Feature::AVX512F);

    // AVX512BW builder can only be used for BlockSize multiples of 512
    if (codegen::BlockSize >= 512 && HasAVX512F) {
        return new KernelBuilderImpl<IDISA_AVX512F_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
    }
    if (codegen::BlockSize >= 256) {
        // AVX2 or AVX builders can only be used for BlockSize multiples of 256
        if (HasAVX2) {
            return new KernelBuilderImpl<IDISA_AVX2_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        } else if (HasAVX) {
            return new KernelBuilderImpl<IDISA_AVX_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
        }
    }
    if (codegen::BlockSize == 128) {
        if (features.lookup("ssse3")) {
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
    return new KernelBuilderImpl<IDISA_Neon_Builder>(C, featureSet, codegen::BlockSize, codegen::LaneWidth);
#elif defined(PARABIX_NVPTX_TARGET)
KernelBuilder * GetIDISA_GPU_Builder(llvm::LLVMContext & C) {
    return new KernelBuilderImpl<IDISA_NVPTX20_Builder>(C, featureSet, 64 * 64, 64);
#else
#error Unknown target type?
#endif
}

}
