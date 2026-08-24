/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

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

template <class F>
llvm::Value *IDISA_SVE_Builder::encapsulateScalableUnary(unsigned fw, llvm::Value *param, F &&createOp) {
    unsigned vectorWidth = getVectorBitWidth(param);
    unsigned fvN = vectorWidth / fw;
    unsigned svN = SVE_min_width / fw;
    unsigned fvChunkN = std::min(vectorWidth / fw, mNativeBitBlockWidth / fw);
    IntegerType *fTy = mCB->getIntNTy(fw);
    FixedVectorType *fvTy = FixedVectorType::get(fTy, fvN);
    FixedVectorType *fvChunkTy = FixedVectorType::get(fTy, fvChunkN);
    ScalableVectorType *svTy = ScalableVectorType::get(fTy, svN);
    ScalableVectorType *svPTy = ScalableVectorType::get(mCB->getInt1Ty(), svN);

    unsigned nChunks = fvN / fvChunkN;

    Value *pred = mCB->CreateIntrinsic(Intrinsic::aarch64_sve_ptrue, {svPTy},
                                       {mCB->getIntN(32, PatternForVectorLength(fvChunkN))});
    Value *result = PoisonValue::get(fvTy);

    param = fwCast(fw, param);
    // In theory, our block width could be bigger than the SVE registers; in that case we must repeat the operation
    // multiple times.
    for (unsigned i = 0; i < nChunks; ++i) {
        // If we're NOT iterating, the input is just the whole fixed param, otherwise extract the appropriate chunk
        Value *inputFixedChunk = (nChunks == 1) ? param
                                                : mCB->CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                       {param, mCB->getIntN(64, i * fvChunkN)});
        // Fixed vector converted to scalable via insert
        Value *inputScalableChunk =
            mCB->CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                 {PoisonValue::get(svTy), inputFixedChunk, mCB->getIntN(64, 0)});
        Value *resultScalableChunk = createOp(svTy, pred, inputScalableChunk);
        // Scalable vector converted to fixed via extract
        Value *resultFixedChunk = mCB->CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, svTy},
                                                       {resultScalableChunk, mCB->getIntN(64, 0)});
        // If we're NOT iterating, result is the fixed chunk directly, otherwise build the full result up in the
        // result value
        result = (nChunks == 1) ? resultFixedChunk
                                : mCB->CreateIntrinsic(Intrinsic::vector_insert, {fvTy, fvChunkTy},
                                                       {result, resultFixedChunk, mCB->getIntN(64, i * fvChunkN)});
    }
    return result;
}

template <class F>
llvm::Value *IDISA_SVE_Builder::encapsulateScalableBinary(unsigned fw, llvm::Value *param1, llvm::Value *param2,
                                                          F &&createOp) {
    unsigned vectorWidth = getVectorBitWidth(param1);
    assert(getVectorBitWidth(param2) == vectorWidth);
    unsigned fvN = vectorWidth / fw;
    unsigned svN = SVE_min_width / fw;
    unsigned fvChunkN = std::min(vectorWidth / fw, mNativeBitBlockWidth / fw);
    IntegerType *fTy = mCB->getIntNTy(fw);
    FixedVectorType *fvTy = FixedVectorType::get(fTy, fvN);
    FixedVectorType *fvChunkTy = FixedVectorType::get(fTy, fvChunkN);
    ScalableVectorType *svTy = ScalableVectorType::get(fTy, svN);
    ScalableVectorType *svPTy = ScalableVectorType::get(mCB->getInt1Ty(), svN);

    unsigned nChunks = fvN / fvChunkN;

    Value *pred = mCB->CreateIntrinsic(Intrinsic::aarch64_sve_ptrue, {svPTy},
                                       {mCB->getIntN(32, PatternForVectorLength(fvChunkN))});
    Value *result = PoisonValue::get(fvTy);

    param1 = fwCast(fw, param1);
    param2 = fwCast(fw, param2);
    // In theory, our block width could be bigger than the SVE registers; in that case we must repeat the operation
    // multiple times.
    for (unsigned i = 0; i < nChunks; ++i) {
        // If we're NOT iterating, the input is just the whole fixed param, otherwise extract the appropriate chunk
        Value *inputFixedChunk1 = (nChunks == 1) ? param1
                                                 : mCB->CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                        {param1, mCB->getIntN(64, i * fvChunkN)});
        Value *inputFixedChunk2 = (nChunks == 1) ? param2
                                                 : mCB->CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, fvTy},
                                                                        {param2, mCB->getIntN(64, i * fvChunkN)});
        // Fixed vector converted to scalable via insert
        Value *inputScalableChunk1 =
            mCB->CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                 {PoisonValue::get(svTy), inputFixedChunk1, mCB->getIntN(64, 0)});
        Value *inputScalableChunk2 =
            mCB->CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvChunkTy},
                                 {PoisonValue::get(svTy), inputFixedChunk2, mCB->getIntN(64, 0)});

        Value *resultScalableChunk = createOp(svTy, pred, inputScalableChunk1, inputScalableChunk2);
        // Scalable vector converted to fixed via extract
        Value *resultFixedChunk = mCB->CreateIntrinsic(Intrinsic::vector_extract, {fvChunkTy, svTy},
                                                       {resultScalableChunk, mCB->getIntN(64, 0)});
        // If we're NOT iterating, result is the fixed chunk directly, otherwise build the full result up in the
        // result value
        result = (nChunks == 1) ? resultFixedChunk
                                : mCB->CreateIntrinsic(Intrinsic::vector_insert, {fvTy, fvChunkTy},
                                                       {result, resultFixedChunk, mCB->getIntN(64, i * fvChunkN)});
    }
    return result;
}

llvm::Value *IDISA_SVE_Builder::simd_fill_impl(unsigned fw, llvm::Value *a) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_fill_impl(fw, a);
}

llvm::Value *IDISA_SVE_Builder::simd_fill_impl(unsigned vector_width, unsigned fw, llvm::Value *a) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_fill_impl(vector_width, fw, a);
}

llvm::Value *IDISA_SVE_Builder::simd_add_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_add, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_add_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_sub_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_sub, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_sub_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_mult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_mul, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_mult_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_eq_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_eq_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_ne_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_ne_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_gt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_gt_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_ge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_ge_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_lt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_lt_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_le_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_le_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_ugt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_ugt_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_ult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_ult_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_ule_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_ule_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_uge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_uge_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_max_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_smax, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_max_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_umax_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umax, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_umax_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_min_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_smin, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_min_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_umin_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableB) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {pred, scalableA, scalableB});
            });
    }
    return IDISA_Generic_Builder::simd_umin_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_if_impl(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_if_impl(fw, cond, a, b);
}

llvm::Value *IDISA_SVE_Builder::simd_ternary_impl(unsigned char mask, llvm::Value *bit_2, llvm::Value *bit_1,
                                                  llvm::Value *bit_0) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_ternary_impl(mask, bit_2, bit_1, bit_0);
}

llvm::Value *IDISA_SVE_Builder::simd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            Value *shiftVec = mCB->CreateIntrinsic(Intrinsic::aarch64_sve_dup_x, {svTy}, {mCB->getIntN(fw, shift)});
            return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_lsl, {svTy}, {pred, scalableA, shiftVec});
        });
    }
    return IDISA_Generic_Builder::simd_slli_impl(fw, a, shift);
}

llvm::Value *IDISA_SVE_Builder::simd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_lsr, {svTy},
                                        {scalableA, pred, scalableA, mCB->getInt32(shift)});
        });
    }
    return IDISA_Generic_Builder::simd_srli_impl(fw, a, shift);
}

llvm::Value *IDISA_SVE_Builder::simd_srai_impl(unsigned fw, llvm::Value *a, unsigned shift) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_asr, {svTy},
                                        {scalableA, pred, scalableA, mCB->getInt32(shift)});
        });
    }
    return IDISA_Generic_Builder::simd_srai_impl(fw, a, shift);
}

llvm::Value *IDISA_SVE_Builder::simd_sllv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, shifts, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableShifts) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_lsl, {svTy},
                                            {scalableA, pred, scalableA, scalableShifts});
            });
    }
    return IDISA_Generic_Builder::simd_sllv_impl(fw, a, shifts);
}

llvm::Value *IDISA_SVE_Builder::simd_srlv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableBinary(
            fw, a, shifts, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value *scalableShifts) {
                return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_lsr, {svTy},
                                            {scalableA, pred, scalableA, scalableShifts});
            });
    }
    return IDISA_Generic_Builder::simd_srlv_impl(fw, a, shifts);
}

llvm::Value *IDISA_SVE_Builder::simd_rotl_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_rotl_impl(fw, a, rotates);
}

llvm::Value *IDISA_SVE_Builder::simd_rotr_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {pred, scalableA});
    //     });
    // }
    return IDISA_Generic_Builder::simd_rotr_impl(fw, a, rotates);
}

llvm::Value *IDISA_SVE_Builder::simd_popcount_impl(unsigned fw, llvm::Value *a) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {PoisonValue::get(svTy), pred, scalableA});
        });
    }
    return IDISA_Generic_Builder::simd_popcount_impl(fw, a);
}

llvm::Value *IDISA_SVE_Builder::simd_bitreverse_impl(unsigned fw, llvm::Value *a) {
    if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        return encapsulateScalableUnary(fw, a, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA) {
            return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_rbit, {svTy}, {PoisonValue::get(svTy), pred, scalableA});
        });
    }
    return IDISA_Generic_Builder::simd_bitreverse_impl(fw, a);
}

llvm::Value *IDISA_SVE_Builder::esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.esimd_mergeh_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.esimd_mergel_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.hsimd_packh_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.hsimd_packl_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.hsimd_packus_impl(fw, a, b);
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                                  ShuffleMode m) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.mvmd_shuffle_impl(fw, data_table, index_vector);
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                                   llvm::Value *index_vector, ShuffleMode m) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, b, [=](ScalableVectorType *svTy, Value *pred, Value *scalableA, Value
    //     *scalableB) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableB});
    //     });
    // }
    return mNeonB.mvmd_shuffle2_impl(fw, table0, table1, index_vector);
}

llvm::Value *IDISA_SVE_Builder::mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    unsigned vectorWidth = getVectorBitWidth(a);
    if ((vectorWidth <= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
        unsigned svN = SVE_min_width / fw;
        IntegerType *fTy = mCB->getIntNTy(fw);
        FixedVectorType *fvTy = FixedVectorType::get(fTy, svN);
        ScalableVectorType *svTy = ScalableVectorType::get(fTy, svN);
        ScalableVectorType *svPTy = ScalableVectorType::get(mCB->getInt1Ty(), svN);

        Type *maskTy = select_mask->getType();
        Value *pred = Constant::getNullValue(svPTy);
        for (unsigned i = 0; i < svN; i++) {
            Value *bit =
                mCB->CreateAnd(mCB->CreateLShr(select_mask, ConstantInt::get(maskTy, i)), ConstantInt::get(maskTy, 1));
            Value *isSet = mCB->CreateICmpNE(bit, ConstantInt::get(maskTy, 0));
            pred = mCB->CreateInsertElement(pred, isSet, mCB->getIntN(64, i));
        }

        Value *scalableA = mCB->CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvTy},
                                                {PoisonValue::get(svTy), fwCast(fw, a), mCB->getIntN(64, 0)});

        Value *compacted = mCB->CreateIntrinsic(Intrinsic::aarch64_sve_compact, {svTy}, {pred, scalableA});
        Value *result = mCB->CreateIntrinsic(Intrinsic::vector_extract, {fvTy, svTy}, {compacted, mCB->getIntN(64, 0)});
        return result;
    } else {
        return mNeonB.mvmd_compress_impl(fw, a, select_mask);
    }
}

llvm::Value *IDISA_SVE_Builder::mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    // if ((getVectorBitWidth(a) >= SVE_min_width) && (fw >= 8) && (fw <= 64)) {
    //     return encapsulateScalableBinary(fw, a, select_mask, [=](ScalableVectorType *svTy, Value *pred, Value
    //     *scalableA, Value *scalableSelectMask) {
    //         return mCB->CreateIntrinsic(Intrinsic::aarch64_sve_umin, {svTy}, {PoisonValue::get(svTy), pred,
    //         scalableA, scalableSelectMask});
    //     });
    // }
    return mNeonB.mvmd_expand_impl(fw, a, select_mask);
}

} // namespace IDISA
