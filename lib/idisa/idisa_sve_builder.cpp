/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "idisa/idisa_builder.h"
#include <idisa/idisa_sve_builder.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace IDISA {

static svpattern PatternForVectorLength(unsigned svN) {
    switch (svN) {
    case 1:
        return SV_VL1;
    case 2:
        return SV_VL2;
    case 4:
        return SV_VL4;
    case 8:
        return SV_VL8;
    case 16:
        return SV_VL16;
    case 32:
        return SV_VL32;
    case 64:
        return SV_VL64;
    case 128:
        return SV_VL128;
    case 256:
        return SV_VL256;
    default:
        report_fatal_error(StringRef("simd_popcount: Vector size has no predicate pattern: ") + std::to_string(svN));
    }
}

unsigned IDISA_SVE_Builder::NativeBitBlockWidth() {
#if 1
    // This unfortunately doesn't work: LLVM doesn't know what to do with big fixed vectors
    return codegen::HostSVEBitWidth();
#else
    return IDISA::ARM_Neon_width;
#endif
}

DEFINE_BUILDER_CACHE_NAME(IDISA_SVE_Builder, "ARM_SVE_VL" + std::to_string(mNativeBitBlockWidth), mNativeBitBlockWidth)

llvm::Value *IDISA_SVE_Builder::simd_popcount(unsigned fw, llvm::Value *a) {
    // TODO JL make a fallback for fw<8 and fw>64..?
    // ScalableVectorType::get(getInt8Ty(), fw, )

    // With SVE it's actually okay to synthesize smaller chunks than "native"
    assert(mNativeBitBlockWidth <= NativeBitBlockWidth());
    assert(mNativeBitBlockWidth >= fw);

    unsigned vectorWidth = getVectorBitWidth(a);
    if ((vectorWidth >= ARM_SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        unsigned fvN = vectorWidth / fw;
        unsigned svN = ARM_SVE_min_width / fw;
        unsigned fvChunkN = std::min(vectorWidth / fw, mNativeBitBlockWidth / fw);
        IntegerType *fTy = getIntNTy(fw);
        FixedVectorType *fvTy = FixedVectorType::get(fTy, fvN);
        FixedVectorType *fvChunkTy = FixedVectorType::get(fTy, fvChunkN);
        ScalableVectorType *svTy = ScalableVectorType::get(fTy, svN);
        ScalableVectorType *svPTy = ScalableVectorType::get(getInt1Ty(), svN);

        unsigned nChunks = fvN / fvChunkN;

        Value *pred =
            CreateIntrinsic(Intrinsic::aarch64_sve_ptrue, {svPTy}, {getIntN(32, PatternForVectorLength(fvChunkN))});
        Value *result = PoisonValue::get(fvTy);

        // In theory, our block width could be bigger than the SVE registers; in that case we must repeat the operation
        // multiple times.
        for (unsigned i = 0; i < nChunks; ++i) {
            // If we're NOT iterating, the input is just the whole fixed a, otherwise extract the appropriate chunk
            Value *inputFixedChunk = (nChunks == 1) ? a
                                                    : CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                      {a, getIntN(64, i * fvChunkN)});
            // Fixed vector converted to scalable via insert
            Value *inputScalableChunk = CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                                        {PoisonValue::get(svTy), inputFixedChunk, getIntN(64, 0)});
            Value *resultScalableChunk =
                CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {PoisonValue::get(svTy), pred, inputScalableChunk});
            // Scalable vector converted to fixed via extract
            Value *resultFixedChunk =
                CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, svTy}, {resultScalableChunk, getIntN(64, 0)});
            // If we're NOT iterating, result is the fixed chunk directly, otherwise build the full result up in the
            // result value
            result = (nChunks == 1) ? resultFixedChunk
                                    : CreateIntrinsic(Intrinsic::vector_insert, {fvTy, fvChunkTy},
                                                      {result, resultFixedChunk, getIntN(64, i * fvChunkN)});
        }
        return result;
    } else {
        return IDISA_Builder::simd_popcount(fw, a);
    }
}

llvm::Value *IDISA_SVE_Builder::simd_bitreverse(unsigned fw, llvm::Value *a) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::simd_bitreverse(fw, a); });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergeh(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::esimd_mergeh(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergel(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::esimd_mergel(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packh(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::hsimd_packh(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::hsimd_packl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packus(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::hsimd_packus(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                             ShuffleMode m) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width,
                             [=]() { return IDISA_ARM_Builder::mvmd_shuffle(fw, data_table, index_vector); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle2(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                              llvm::Value *index_vector, ShuffleMode m) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width,
                             [=]() { return IDISA_ARM_Builder::mvmd_shuffle2(fw, table0, table1, index_vector); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_compress(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::mvmd_compress(fw, a, select_mask); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_expand(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::mvmd_expand(fw, a, select_mask); });
}

} // namespace IDISA
