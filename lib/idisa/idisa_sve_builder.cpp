/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "idisa/idisa_neon_builder.h"
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

DEFINE_BUILDER_CACHE_NAME(IDISA_SVE_Builder, "ARM_SVE_VL" + std::to_string(getBitBlockWidth()), getBitBlockWidth())

#if 1
IDISA_SVE_Builder::IDISA_SVE_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth)
    : IDISA::IDISA_Generic_Builder(cb, vectorWidth, laneWidth, codegen::HostSVEBitWidth()),
      mNeonB(cb, vectorWidth, laneWidth) {}
#else
IDISA_SVE_Builder::IDISA_SVE_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth)
    : IDISA::IDISA_Generic_Builder(cb, vectorWidth, laneWidth, SVE_min_width), mNeonB(cb, vectorWidth, laneWidth) {}
#endif

template <class F> llvm::Value *IDISA_SVE_Builder::withNativeWidth_impl(unsigned tempWidth, F &&f) {
    unsigned realWidth = mNativeBitBlockWidth;
    try {
        const_cast<unsigned &>(mNativeBitBlockWidth) = tempWidth;
        llvm::Value *result = f();
        const_cast<unsigned &>(mNativeBitBlockWidth) = realWidth;
        return result;
    } catch (...) {
        const_cast<unsigned &>(mNativeBitBlockWidth) = realWidth;
        throw;
    }
}

template <class F>
llvm::Value *IDISA_SVE_Builder::encapsulateScalableUnary_impl(unsigned fw, llvm::Value *param, F &&createOp) {
    unsigned vectorWidth = getVectorBitWidth(param);
    unsigned fvN = vectorWidth / fw;
    unsigned svN = SVE_min_width / fw;
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
        // If we're NOT iterating, the input is just the whole fixed param, otherwise extract the appropriate chunk
        Value *inputFixedChunk = (nChunks == 1) ? param
                                                : CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                  {param, getIntN(64, i * fvChunkN)});
        // Fixed vector converted to scalable via insert
        Value *inputScalableChunk = CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                                    {PoisonValue::get(svTy), inputFixedChunk, getIntN(64, 0)});
        Value *resultScalableChunk = createOp(svTy, pred, inputScalableChunk);
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
}

template <class F>
llvm::Value *IDISA_SVE_Builder::encapsulateScalableBinary_impl(unsigned fw, llvm::Value *param1, llvm::Value *param2,
                                                               F &&createOp) {
    unsigned vectorWidth = getVectorBitWidth(param1);
    assert(getVectorBitWidth(param2) == vectorWidth);
    unsigned fvN = vectorWidth / fw;
    unsigned svN = SVE_min_width / fw;
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
        // If we're NOT iterating, the input is just the whole fixed param, otherwise extract the appropriate chunk
        Value *inputFixedChunk1 = (nChunks == 1) ? param1
                                                 : CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                   {param1, getIntN(64, i * fvChunkN)});
        Value *inputFixedChunk2 = (nChunks == 1) ? param2
                                                 : CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                   {param2, getIntN(64, i * fvChunkN)});
        // Fixed vector converted to scalable via insert
        Value *inputScalableChunk1 = CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                                     {PoisonValue::get(svTy), inputFixedChunk1, getIntN(64, 0)});
        Value *inputScalableChunk2 = CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                                     {PoisonValue::get(svTy), inputFixedChunk2, getIntN(64, 0)});

        Value *resultScalableChunk = createOp(svTy, pred, inputScalableChunk1, inputScalableChunk2);
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
}

llvm::Value *IDISA_SVE_Builder::simd_popcount_impl(unsigned fw, llvm::Value *a) {
    unsigned vectorWidth = getVectorBitWidth(a);
    if ((vectorWidth >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            return CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {PoisonValue::get(svTy), pred, scalableA});
        });
    } else {
        return IDISA_Builder::simd_popcount(fw, a);
    }
}

llvm::Value *IDISA_SVE_Builder::simd_bitreverse_impl(unsigned fw, llvm::Value *a) {
    unsigned vectorWidth = getVectorBitWidth(a);
    if ((vectorWidth >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            return CreateIntrinsic(Intrinsic::aarch64_sve_rbit, {svTy}, {PoisonValue::get(svTy), pred, scalableA});
        });
    } else {
        return IDISA_Builder::simd_bitreverse(fw, a);
    }
}

llvm::Value *IDISA_SVE_Builder::esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // unsigned vectorWidth = getVectorBitWidth(a);
    // if ((vectorWidth >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return CreateIntrinsic(Intrinsic::aarch64_sve_rbit, {svTy}, {PoisonValue::get(svTy), pred, scalableA});
    //     });
    // } else {
    //     return IDISA_Builder::simd_bitreverse(fw, a);
    // }
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::esimd_mergeh_impl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::esimd_mergel_impl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::hsimd_packh_impl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::hsimd_packl_impl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::hsimd_packus_impl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                                  ShuffleMode m) {
    // TODO JL implement
    return withNativeWidth(Neon_width,
                           [=]() { return IDISA_Neon_Builder::mvmd_shuffle_impl(fw, data_table, index_vector); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle2(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                              llvm::Value *index_vector, ShuffleMode m) {
    // TODO JL implement
    return withNativeWidth(Neon_width,
                           [=]() { return IDISA_Neon_Builder::mvmd_shuffle2(fw, table0, table1, index_vector); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::mvmd_compress_impl(fw, a, select_mask); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    return withNativeWidth(Neon_width, [=]() { return IDISA_Neon_Builder::mvmd_expand_impl(fw, a, select_mask); });
}

} // namespace IDISA
