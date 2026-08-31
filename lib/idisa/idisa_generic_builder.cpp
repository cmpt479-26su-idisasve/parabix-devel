/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_generic_builder.h>

#include <boost/intrusive/detail/math.hpp>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <pthread.h>
#include <toolchain/toolchain.h>
#include <unistd.h>

using boost::intrusive::detail::floor_log2;

// #define PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM

using namespace llvm;

namespace IDISA {

// getLaneWidth() gives the largest field width we can do integer/bit manipulation over, and sometimes that is wider
// than is really efficient -- bitManipFW is a width that's supposed to be efficient, i.e. it has add/and/or/not/etc.
// and splatted 8-bit constants all available and efficient.

// **Note that 8 bits is probably too small!** We default to 32 here because that works for most architectures. If this
// is made variable somehow, care should be taken: this field size may be used for math on field and bit indices, so
// it's important that a uint of this width can hold a bit index for the whole SIMD vector. That is, if we're dealing
// with 512-bit SIMD vector, and bitManipFW is 8, values will be restriced to 0-255 which obviously doesn't cover the
// whole 0-511 range. So on that machine, bitManipFW has to be 16 at least.
constexpr unsigned bitManipFW = 32;

Value *IDISA_Generic_Builder::simd_fill_impl(unsigned fw, Value *a) { return simd_fill(getBitBlockWidth(), fw, a); }

Value *IDISA_Generic_Builder::simd_fill_impl(unsigned vec_width, unsigned fw, Value *a) {
    if (fw < 8) {
        assert((fw & (fw - 1)) == 0);
        assert((vec_width % 8) == 0);
        Type *vecTy = FixedVectorType::get(mCB->getIntNTy(fw), vec_width / fw);
        if (fw == 1) {
            // sign extend single bit
            return mCB->CreateBitCast(
                simd_fill(vec_width, bitManipFW,
                          mCB->CreateSExt(mCB->CreateTrunc(a, mCB->getInt1Ty()), mCB->getIntNTy(bitManipFW))),
                vecTy);
        } else {
            Value *extendA = mCB->CreateZExtOrTrunc(a, mCB->getInt8Ty());
            if (fw == 2) {
                extendA = mCB->CreateOr(extendA, mCB->CreateShl(extendA, 2));
            }
            extendA = mCB->CreateOr(extendA, mCB->CreateShl(extendA, 4));
            return mCB->CreateBitCast(simd_fill(vec_width, 8, extendA), vecTy);
        }
    }
    const unsigned field_count = vec_width / fw;
    Type *singleFieldVecTy = FixedVectorType::get(mCB->getIntNTy(fw), 1);
    Value *aVec = mCB->CreateBitCast(mCB->CreateZExtOrTrunc(a, mCB->getIntNTy(fw)), singleFieldVecTy);
    return mCB->CreateShuffleVector(aVec, UndefValue::get(singleFieldVecTy),
                                    Constant::getNullValue(FixedVectorType::get(mCB->getInt32Ty(), field_count)));
}

Value *IDISA_Generic_Builder::simd_add_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1) {
        return simd_xor(a, b);
    } else if (fw < 8) {
        const unsigned vectorWidth = getVectorBitWidth(a);
        Constant *hi_bit_mask = getSplatN(fw, vectorWidth / fw, 1 << (fw - 1));
        Constant *lo_bit_mask = getSplatN(fw, vectorWidth / fw, (1 << (fw - 1)) - 1);
        Value *hi_xor = simd_xor(simd_and(a, hi_bit_mask), simd_and(b, hi_bit_mask));
        Value *part_sum = simd_add(bitManipFW, simd_and(a, lo_bit_mask), simd_and(b, lo_bit_mask));
        return simd_xor(part_sum, hi_xor);
    }
    return mCB->CreateAdd(fwCast(fw, a), fwCast(fw, b));
}

Value *IDISA_Generic_Builder::simd_sub_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1) {
        return simd_xor(a, b);
    } else if (fw < 8) {
        const unsigned vectorWidth = getVectorBitWidth(a);
        Constant *ones = getSplatN(fw, vectorWidth / fw, 1);
        Constant *hi_bit_mask = getSplatN(fw, vectorWidth / fw, 1 << (fw - 1));
        Constant *lo_bit_mask = getSplatN(fw, vectorWidth / fw, (1 << (fw - 1)) - 1);
        Value *not_b = simd_not(b);
        Value *hi_xor = simd_xor(simd_and(a, hi_bit_mask), simd_and(not_b, hi_bit_mask));
        Value *part_sum =
            simd_add(bitManipFW, simd_add(bitManipFW, simd_and(a, lo_bit_mask), simd_and(not_b, lo_bit_mask)), ones);
        return simd_xor(part_sum, hi_xor);
    }
    return mCB->CreateSub(fwCast(fw, a), fwCast(fw, b));
}

Value *IDISA_Generic_Builder::simd_mult_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1) {
        return simd_and(a, b);
    } else if (fw == 2) {
        Value *hi_b_mask = simd_select_hi(fw, b);
        Value *lo_b_mask = simd_select_lo(fw, b);
        lo_b_mask = simd_or(lo_b_mask, simd_slli(bitManipFW, lo_b_mask, 1));
        Value *hi_a = simd_slli(bitManipFW, a, 1);
        return simd_add(fw, simd_and(a, lo_b_mask), simd_and(hi_a, hi_b_mask));
    } else if (fw == 4) {
        // Do 4-bit multiply using 2 8-bit multiplies
        // We do the 8-bit multiply with junk in the top nybbles. This only affects output top nybbles.
        // The subsequent selects mask off the junk.
        Value *bot_prod = simd_select_lo(8, simd_mult(8, a, b));
        Value *top_prod = simd_select_lo(8, simd_mult(8, simd_srli(bitManipFW, a, 4), simd_srli(bitManipFW, b, 4)));
        return simd_or(simd_slli(bitManipFW, simd_select_lo(8, top_prod), 4), simd_select_lo(8, bot_prod));
    }
    return mCB->CreateMul(fwCast(fw, a), fwCast(fw, b));
}

Value *IDISA_Generic_Builder::simd_eq_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 8) {
        Value *eq_bits = simd_not(simd_xor(a, b));
        if (fw == 1)
            return eq_bits;
        eq_bits = simd_or(simd_and(simd_srli(bitManipFW, simd_select_hi(2, eq_bits), 1), eq_bits),
                          simd_and(simd_slli(bitManipFW, simd_select_lo(2, eq_bits), 1), eq_bits));
        if (fw == 2)
            return eq_bits;
        eq_bits = simd_or(simd_and(simd_srli(bitManipFW, simd_select_hi(4, eq_bits), 2), eq_bits),
                          simd_and(simd_slli(bitManipFW, simd_select_lo(4, eq_bits), 2), eq_bits));
        return eq_bits;
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpEQ(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_any_impl(unsigned fw, Value *a) {
    return mCB->CreateNot(simd_eq(fw, a, ConstantVector::getNullValue(a->getType())));
}

Value *IDISA_Generic_Builder::simd_ne_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 8) {
        Value *ne_bits = simd_xor(a, b);
        if (fw == 1)
            return ne_bits;
        ne_bits = simd_or(simd_or(simd_srli(bitManipFW, simd_select_hi(2, ne_bits), 1), ne_bits),
                          simd_or(simd_slli(bitManipFW, simd_select_lo(2, ne_bits), 1), ne_bits));
        if (fw == 2)
            return ne_bits;
        ne_bits = simd_or(simd_or(simd_srli(bitManipFW, simd_select_hi(4, ne_bits), 2), ne_bits),
                          simd_or(simd_slli(bitManipFW, simd_select_lo(4, ne_bits), 2), ne_bits));
        return ne_bits;
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpNE(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_gt_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_and(simd_not(a), b);
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_gt(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_gt(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpSGT(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_ge_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_or(simd_not(a), b);
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_ge(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_ge(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpSGE(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_ugt_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_and(a, simd_not(b));
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_ugt(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_ugt(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpUGT(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_lt_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_and(a, simd_not(b));
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_lt(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_lt(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpSLT(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_le_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_or(a, simd_not(b));
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_le(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_le(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpSLE(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_ult_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_and(simd_not(a), b);
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_ult(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_ult(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpULT(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_ule_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_or(simd_not(a), b);
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_ule(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_ule(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpULE(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_uge_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_or(a, simd_not(b));
    if (fw < 8) {
        Value *hi_rslt = simd_select_hi(2 * fw, simd_uge(2 * fw, simd_select_hi(2 * fw, a), simd_select_hi(2 * fw, b)));
        Value *lo_rslt = simd_select_lo(2 * fw, simd_uge(2 * fw, simd_slli(2 * fw, a, fw), simd_slli(2 * fw, b, fw)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value *a1 = fwCast(fw, a);
    Value *b1 = fwCast(fw, b);
    return mCB->CreateSExt(mCB->CreateICmpUGE(a1, b1), a1->getType());
}

Value *IDISA_Generic_Builder::simd_max_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_and(a, b);
    if (fw < 8) {
        Value *test = simd_gt(fw, a, b);
        return simd_or(simd_and(test, a), simd_and(simd_not(test), b));
    }
    Value *aVec = fwCast(fw, a);
    Value *bVec = fwCast(fw, b);
    return mCB->CreateSelect(mCB->CreateICmpSGT(aVec, bVec), aVec, bVec);
}

Value *IDISA_Generic_Builder::simd_umax_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_or(a, b);
    if (fw < 8) {
        Value *test = simd_ugt(fw, a, b);
        return simd_or(simd_and(test, a), simd_and(simd_not(test), b));
    }
    Value *aVec = fwCast(fw, a);
    Value *bVec = fwCast(fw, b);
    return mCB->CreateSelect(mCB->CreateICmpUGT(aVec, bVec), aVec, bVec);
}

Value *IDISA_Generic_Builder::simd_min_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_or(a, b);
    if (fw < 8) {
        Value *test = simd_lt(fw, a, b);
        return simd_or(simd_and(test, a), simd_and(simd_not(test), b));
    }
    Value *aVec = fwCast(fw, a);
    Value *bVec = fwCast(fw, b);
    return mCB->CreateSelect(mCB->CreateICmpSLT(aVec, bVec), aVec, bVec);
}

Value *IDISA_Generic_Builder::simd_umin_impl(unsigned fw, Value *a, Value *b) {
    if (fw == 1)
        return simd_and(a, b);
    if (fw < 8) {
        Value *test = simd_ult(fw, a, b);
        return simd_or(simd_and(test, a), simd_and(simd_not(test), b));
    }
    Value *aVec = fwCast(fw, a);
    Value *bVec = fwCast(fw, b);
    return mCB->CreateSelect(mCB->CreateICmpULT(aVec, bVec), aVec, bVec);
}

Value *IDISA_Generic_Builder::mvmd_sll_impl(unsigned fw, Value *a, Value *shift, const bool safe) {
    unsigned vec_width = getVectorBitWidth(a);
    Type *shiftTy = shift->getType();
    if (LLVM_UNLIKELY(!shiftTy->isIntegerTy())) {
        report_fatal_error("shift value type must be an integer");
    }
    Type *sizeTy = mCB->getSizeTy();
    if (shiftTy != sizeTy) {
        shift = mCB->CreateZExtOrTrunc(shift, sizeTy);
    }
    shift = mCB->CreateMul(shift, mCB->getSize(fw));
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value *const inbounds = mCB->CreateICmpULT(shift, mCB->getSize(vec_width));
        mCB->CreateAssert(inbounds, "poison shift value: >= vector width");
    }

    IntegerType *const intTy = mCB->getIntNTy(vec_width);
    Value *result = nullptr;

    a = mCB->CreateBitCast(a, intTy);
    //    if (safe) {
    shift = mCB->CreateZExtOrTrunc(shift, intTy);
    result = mCB->CreateShl(a, shift);
    //    } else {
    //        // TODO: check the ASM generated by this to see what the select generates
    //        Value * const moddedShift = CreateURem(shift, BLOCK_WIDTH);
    //        Value * const inbounds = CreateICmpEQ(moddedShift, shift);
    //        shift = CreateZExtOrTrunc(moddedShift, intTy);
    //        Constant * const ZEROES = Constant::getNullValue(intTy);
    //        result = CreateShl(value, shift);
    //        result = CreateSelect(inbounds, result, ZEROES);
    //    }
    return result;
}

Value *IDISA_Generic_Builder::mvmd_dsll_impl(unsigned fw, Value *a, Value *b, Value *shift) {
    if (fw < 8)
        UnsupportedFieldWidthError(fw, "mvmd_dsll");
    unsigned vec_width = getVectorBitWidth(a);
    const auto field_count = vec_width / fw;
    Type *fwTy = mCB->getIntNTy(fw);
    IndexVector Idxs(field_count);
    for (unsigned i = 0; i < field_count; i++) {
        Idxs[i] = ConstantInt::get(fwTy, i + field_count);
    }
    Value *shuffle_indexes = simd_sub(fw, ConstantVector::get(Idxs), simd_fill(vec_width, fw, shift));
    return mvmd_shuffle2(fw, b, a, shuffle_indexes);
}

Value *IDISA_Generic_Builder::mvmd_srl_impl(unsigned fw, Value *a, Value *shift, const bool safe) {
    unsigned vec_width = getVectorBitWidth(a);
    Type *shiftTy = shift->getType();
    if (LLVM_UNLIKELY(!shiftTy->isIntegerTy())) {
        report_fatal_error("shift value type must be an integer");
    }
    Type *sizeTy = mCB->getSizeTy();
    if (shiftTy != sizeTy) {
        shift = mCB->CreateZExtOrTrunc(shift, sizeTy);
    }
    shift = mCB->CreateMul(shift, mCB->getSize(fw));
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value *const inbounds = mCB->CreateICmpULT(shift, mCB->getSize(vec_width));
        mCB->CreateAssert(inbounds, "poison shift value: >= vector width");
    }

    IntegerType *const intTy = mCB->getIntNTy(vec_width);
    Value *result = nullptr;

    a = mCB->CreateBitCast(a, intTy);
    //    if (safe) {
    shift = mCB->CreateZExtOrTrunc(shift, intTy);
    result = mCB->CreateLShr(a, shift);
    //    } else {
    //        // TODO: check the ASM generated by this to see what the select generates
    //        Value * const moddedShift = CreateURem(shift, BLOCK_WIDTH);
    //        Value * const inbounds = CreateICmpEQ(moddedShift, shift);
    //        shift = CreateZExtOrTrunc(moddedShift, intTy);
    //        Constant * const ZEROES = Constant::getNullValue(intTy);
    //        result = CreateLShr(value, shift);
    //        result = CreateSelect(inbounds, result, ZEROES);
    //    }
    return result;
}

Value *IDISA_Generic_Builder::simd_slli_impl(unsigned fw, Value *a, unsigned shift) {
    assert(shift < fw);
    if (shift == 0)
        return a;
    const unsigned vectorWidth = getVectorBitWidth(a);
    if (fw > mMaxNativeSimdShift) {
        unsigned fullFieldShift = shift / mMaxNativeSimdShift;
        unsigned subFieldShift = shift % mMaxNativeSimdShift;
        Value *fieldShifted = a;
        if (fullFieldShift > 0) {
            fieldShifted = mvmd_slli(mMaxNativeSimdShift, a, fullFieldShift);
            unsigned remaining = fw - fullFieldShift * mMaxNativeSimdShift;
            Constant *mask = Constant::getIntegerValue(
                mCB->getIntNTy(fw), APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw, remaining)));
            fieldShifted = simd_and(fieldShifted, mask);
        }
        if (subFieldShift == 0)
            return fieldShifted;
        Value *extendedShift = simd_slli(mMaxNativeSimdShift, fieldShifted, subFieldShift);
        if (fw - fullFieldShift < mMaxNativeSimdShift) {
            // No additional bits to combined
            return extendedShift;
        }
        Value *overShifted = mvmd_slli(mMaxNativeSimdShift, a, fullFieldShift + 1);
        unsigned backShift = mMaxNativeSimdShift - subFieldShift;
        Value *backShifted = simd_srli(mMaxNativeSimdShift, overShifted, backShift);
        return simd_or(extendedShift, backShifted);
    }
    if (fw < mMinNativeSimdShift) {
        Constant *value_mask = Constant::getIntegerValue(
            mCB->getIntNTy(vectorWidth), APInt::getSplat(vectorWidth, APInt::getLowBitsSet(fw, fw - shift)));
        return mCB->CreateShl(fwCast(mMinNativeSimdShift, simd_and(a, value_mask)), shift);
    }
    return mCB->CreateShl(fwCast(fw, a), shift);
}

Value *IDISA_Generic_Builder::simd_srli_impl(unsigned fw, Value *a, unsigned shift) {
    assert(shift < fw);
    if (shift == 0)
        return a;
    const unsigned vectorWidth = getVectorBitWidth(a);
    if (fw > mMaxNativeSimdShift) {
        unsigned fullFieldShift = shift / mMaxNativeSimdShift;
        unsigned subFieldShift = shift % mMaxNativeSimdShift;
        Value *fieldShifted = a;
        if (fullFieldShift > 0) {
            fieldShifted = mvmd_srli(mMaxNativeSimdShift, a, fullFieldShift);
            unsigned remaining = fw - fullFieldShift * mMaxNativeSimdShift;
            Constant *mask = Constant::getIntegerValue(
                mCB->getIntNTy(fw), APInt::getSplat(vectorWidth, APInt::getLowBitsSet(fw, remaining)));
            fieldShifted = simd_and(fieldShifted, mask);
        }
        if (subFieldShift == 0)
            return fieldShifted;
        Value *extendedShift = simd_srli(mMaxNativeSimdShift, fieldShifted, subFieldShift);
        if (fw - fullFieldShift < mMaxNativeSimdShift) {
            // No additional bits to combined
            return extendedShift;
        }
        Value *overShifted = mvmd_srli(mMaxNativeSimdShift, a, fullFieldShift + 1);
        unsigned backShift = mMaxNativeSimdShift - subFieldShift;
        Value *backShifted = simd_slli(mMaxNativeSimdShift, overShifted, backShift);
        return simd_or(extendedShift, backShifted);
    }
    if (fw < mMinNativeSimdShift) {
        Constant *value_mask = Constant::getIntegerValue(
            mCB->getIntNTy(vectorWidth), APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw, fw - shift)));
        return mCB->CreateLShr(fwCast(mMinNativeSimdShift, simd_and(a, value_mask)), shift);
    }
    return mCB->CreateLShr(fwCast(fw, a), shift);
}

Value *IDISA_Generic_Builder::simd_srai_impl(unsigned fw, Value *a, unsigned shift) {
    assert(shift < fw);
    if (shift == 0)
        return a;
    if (fw < mMinNativeSimdShift) {
        // LLVM implicitly widens fw smaller than 8, if mMinNativeSimdShift is small it's not clear what happens
        assert(mMinNativeSimdShift >= 8);
        Value *sign = mCB->CreateAnd(fwCast(mMinNativeSimdShift, a),
                                     APInt::getSplat(mMinNativeSimdShift, APInt::getHighBitsSet(fw, 1)));
        for (unsigned i = 1; i < shift; i *= 2) {
            sign = simd_or(sign, simd_srli(mMinNativeSimdShift, sign, std::min(i, shift - i)));
        }
        return simd_or(sign,
                       mCB->CreateAnd(fwCast(mMinNativeSimdShift, simd_srli(mMinNativeSimdShift, a, shift)),
                                      APInt::getSplat(mMinNativeSimdShift, APInt::getLowBitsSet(fw, fw - shift))));
    }
    return mCB->CreateAShr(fwCast(fw, a), shift);
}

Value *IDISA_Generic_Builder::simd_sllv_impl(unsigned fw, Value *v, Value *shifts) {
    if (fw == 1) {
        return simd_and(v, simd_not(shifts));
    }
    auto vec_width = getVectorBitWidth(v);
    if ((fw == 2 || fw == 4)) {
        auto splat8 = [&](uint8_t x) { return getSplat(vec_width / 8, mCB->getInt8(x)); };
        v = fwCast(8, v);
        shifts = fwCast(8, shifts);
        if (fw == 4) {
            // remask each nibble after the byte shift so bits never carry across the nibble boundary
            Value *loAmt = simd_and(shifts, splat8(0x03));
            Value *hiAmt = simd_srli(8, shifts, 4);
            Value *loSh = simd_and(simd_sllv(8, v, loAmt), splat8(0x0F));
            Value *hiSh = simd_sllv(8, simd_and(v, splat8(0xF0)), hiAmt);
            return simd_or(loSh, hiSh);
        } else {
            // fw == 2: amount is one bit per field; expand it to a full 0b11 field mask and BSL-select
            Value *shifted = simd_and(mCB->CreateShl(fwCast(8, v), splat8(1)), splat8(0xAA));
            Value *a = simd_and(shifts, splat8(0x55));
            Value *sel = simd_or(a, mCB->CreateShl(fwCast(8, a), splat8(1)));
            return simd_or(simd_and(shifted, sel), simd_and(v, simd_not(sel)));
        }
    }
    return mCB->CreateShl(fwCast(fw, v), fwCast(fw, shifts));
}

Value *IDISA_Generic_Builder::simd_srlv_impl(unsigned fw, Value *v, Value *shifts) {
    if (fw == 1) {
        return simd_and(v, simd_not(shifts));
    }
    auto vec_width = getVectorBitWidth(v);
    if ((fw == 2 || fw == 4)) {
        auto splat8 = [&](uint8_t x) { return getSplat(vec_width / 8, mCB->getInt8(x)); };
        v = fwCast(8, v);
        shifts = fwCast(8, shifts);
        if (fw == 4) {
            Value *loAmt = simd_and(shifts, splat8(0x03));
            Value *hiAmt = simd_srli(8, shifts, 4);
            Value *loSh = simd_srlv(8, simd_and(v, splat8(0x0F)), loAmt);
            Value *hiSh = simd_and(simd_srlv(8, v, hiAmt), splat8(0xF0));
            return simd_or(loSh, hiSh);
        } else {
            Value *shifted = simd_and(simd_srli(8, v, 1), splat8(0x55));
            Value *a = simd_and(shifts, splat8(0x55));
            Value *sel = simd_or(a, simd_slli(8, a, 1));
            return simd_or(simd_and(shifted, sel), simd_and(v, simd_not(sel)));
        }
    }
    return mCB->CreateLShr(fwCast(fw, v), fwCast(fw, shifts));
}

Value *IDISA_Generic_Builder::simd_rotl_impl(unsigned fw, Value *v, Value *rotates) {
    if (fw == 1) {
        return v;
    }
    Type *fwTy = mCB->getIntNTy(fw);
    unsigned numFields = getVectorBitWidth(v) / fw;
    Constant *fw_mask = getSplat(numFields, ConstantInt::get(fwTy, fw - 1));
    Constant *fw_splat = getSplat(numFields, ConstantInt::get(fwTy, fw));
    Value *shft = simd_and(fw_mask, rotates);
    Value *fwd = simd_sllv(fw, v, shft);
    // Masking is necessary to avoid a srlv by fw (poison value) when the
    // rotate amount is 0.
    Value *back = simd_srlv(fw, v, simd_and(fw_mask, simd_sub(fw, fw_splat, shft)));
    return simd_or(fwd, back);
}

Value *IDISA_Generic_Builder::simd_rotr_impl(unsigned fw, Value *v, Value *rotates) {
    if (fw == 1) {
        return v;
    }
    Type *fwTy = mCB->getIntNTy(fw);
    unsigned numFields = getVectorBitWidth(v) / fw;
    Constant *fw_mask = getSplat(numFields, ConstantInt::get(fwTy, fw - 1));
    Constant *fw_splat = getSplat(numFields, ConstantInt::get(fwTy, fw));
    Value *shft = simd_and(fw_mask, rotates);
    // Masking is necessary to avoid a sllv by fw (poison value) when the
    // rotate amount is 0.
    Value *fwd = simd_sllv(fw, v, simd_and(fw_mask, simd_sub(fw, fw_splat, shft)));
    Value *back = simd_srlv(fw, v, shft);
    return simd_or(fwd, back);
}

std::vector<Value *> IDISA_Generic_Builder::simd_pext_impl(unsigned fieldwidth, std::vector<Value *> v,
                                                           Value *extract_mask) {
    if (fieldwidth == 1) {
        std::vector<Value *> w;
        for (Value *vv : v) {
            w.emplace_back(simd_and(vv, extract_mask));
        }
        return w;
    }

    Value *delcounts = mCB->CreateNot(extract_mask); // initially deletion counts per 1-bit field
    std::vector<Value *> w(v.size());
    for (unsigned i = 0; i < v.size(); i++) {
        w[i] = simd_and(extract_mask, v[i]);
    }
    for (unsigned fw = 2; fw < fieldwidth; fw = fw * 2) {
        Value *shift_fwd_amts = simd_srli(fw, simd_select_lo(fw * 2, delcounts), fw / 2);
        Value *shift_back_amts = simd_select_lo(fw, simd_select_hi(fw * 2, delcounts));
        for (unsigned i = 0; i < v.size(); i++) {
            w[i] = simd_or(simd_sllv(fw, simd_select_lo(fw * 2, w[i]), shift_fwd_amts),
                           simd_srlv(fw, simd_select_hi(fw * 2, w[i]), shift_back_amts));
        }
        delcounts = simd_add(fw, simd_select_lo(fw, delcounts), simd_srli(fw, delcounts, fw / 2));
    }
    // Now shift back all fw fields.
    Value *shift_back_amts = simd_select_lo(fieldwidth, delcounts);
    for (unsigned i = 0; i < v.size(); i++) {
        w[i] = simd_srlv(fieldwidth, w[i], shift_back_amts);
    }
    return w;
}

Value *IDISA_Generic_Builder::simd_pdep_impl(unsigned fieldwidth, Value *v, Value *deposit_mask) {
    if (fieldwidth == 1) {
        return simd_and(v, deposit_mask);
    }

    // simd_pdep is implemented by reversing the process of simd_pext.
    // First determine the deletion counts necessary for each stage of the process.
    std::vector<Value *> delcounts;
    delcounts.push_back(simd_not(deposit_mask)); // initially deletion counts per 1-bit field
    for (unsigned fw = 2; fw < fieldwidth; fw = fw * 2) {
        delcounts.push_back(
            simd_add(fw, simd_select_lo(fw, delcounts.back()), simd_srli(fw, delcounts.back(), fw / 2)));
    }
    //
    // Now reverse the pext process.  First reverse the final shift_back.
    Value *pext_shift_back_amts = simd_select_lo(fieldwidth, delcounts.back());
    Value *w = simd_sllv(fieldwidth, v, pext_shift_back_amts);
    //
    // No work through the smaller field widths.
    for (unsigned fw = fieldwidth / 2; fw >= 2; fw = fw / 2) {
        delcounts.pop_back();
        Value *pext_shift_fwd_amts = simd_srli(fw, simd_select_lo(fw * 2, delcounts.back()), fw / 2);
        Value *pext_shift_back_amts = simd_select_lo(fw, simd_select_hi(fw * 2, delcounts.back()));
        w = simd_or(simd_srlv(fw, simd_select_lo(fw * 2, w), pext_shift_fwd_amts),
                    simd_sllv(fw, simd_select_hi(fw * 2, w), pext_shift_back_amts));
    }
    return simd_and(w, deposit_mask);
}

Value *IDISA_Generic_Builder::simd_popcount_impl(unsigned fw, Value *a) {
    if (fw == 1) {
        return a;
    } else if (fw == 2) {
        // For each 2-bit field ab we can use the subtraction ab - 0a to generate
        // the popcount without carry/borrow from the neighbouring 2-bit field.
        // case 00:  ab - 0a = 00 - 00 = 00
        // case 01:  ab - 0a = 01 - 00 = 01
        // case 10:  ab - 0a = 10 - 01 = 01 (no borrow)
        // case 11:  ab - 0a = 11 - 01 = 10
        return simd_sub(64, a, simd_srli(64, simd_select_hi(2, a), 1));
    } else if (fw <= 64) {
        Value *c = simd_popcount(fw / 2, a);
        c = simd_add(64, simd_select_lo(fw, c), simd_srli(fw, c, fw / 2));
        return c;
    } else {
        return mCB->CreatePopcount(fwCast(fw, a));
    }
}

Value *IDISA_Generic_Builder::hsimd_partial_sum_impl(unsigned fw, Value *a) {
    const unsigned vectorWidth = getVectorBitWidth(a);
    Value *partial_sum = fwCast(fw, a);
    const auto count = vectorWidth / fw;
    for (unsigned move = 1; move < count; move *= 2) {
        partial_sum = simd_add(fw, partial_sum, mvmd_slli(fw, partial_sum, move));
    }
    return partial_sum;
}

Value *IDISA_Generic_Builder::simd_cttz_impl(unsigned fw, Value *a) {
    if (fw == 1) {
        return simd_not(a);
    } else {
        Value *v = simd_sub(fw, a, simd_fill(fw, mCB->getIntN(fw, 1)));
        v = simd_or(v, a);
        v = simd_xor(v, a);
        v = simd_popcount(fw, v);
        return v;
    }
}

Value *IDISA_Generic_Builder::simd_bitreverse_impl(unsigned fw, Value *a) {
    /*  Pure sequential solution too slow!
     Function * func = Intrinsic::getDeclaration(getModule(), Intrinsic::bitreverse, fwVectorType(fw));
     return CreateCall(func->getFunctionType(), func, fwCast(fw, a));
     */
    if (fw > 8) {
        // Reverse the bits of each byte and then use a byte shuffle to complete the job.
        Value *bitrev8 = fwCast(8, simd_bitreverse(8, a));
        const auto bytes_per_field = fw / 8;
        const unsigned vectorWidth = getVectorBitWidth(a);
        const auto byte_count = vectorWidth / 8;
        IndexVector Idxs(byte_count);
        for (unsigned i = 0; i < byte_count; i += bytes_per_field) {
            for (unsigned j = 0; j < bytes_per_field; j++) {
                Idxs[i + j] = mCB->getInt32(i + bytes_per_field - j - 1);
            }
        }
        return mCB->CreateShuffleVector(bitrev8, UndefValue::get(bitrev8->getType()), ConstantVector::get(Idxs));
    } else {
        if (fw > 2) {
            a = simd_bitreverse(fw / 2, a);
        }
        return simd_or(simd_srli(16, simd_select_hi(fw, a), fw / 2), simd_slli(16, simd_select_lo(fw, a), fw / 2));
    }
}

Value *IDISA_Generic_Builder::simd_if_impl(unsigned fw, Value *cond, Value *a, Value *b) {
    if (fw < 8) {
        // simd_srai(..., 0) generates a no-op when fw=1
        Value *c = simd_srai(fw, cond, fw - 1);
        return simd_or(simd_and(a, c), simd_and(simd_xor(c, b), b));
    } else {
        Value *aVec = fwCast(fw, a);
        Value *bVec = fwCast(fw, b);
        return mCB->CreateSelect(mCB->CreateICmpSLT(fwCast(fw, cond), ConstantVector::getNullValue(aVec->getType())),
                                 aVec, bVec);
    }
}

Value *IDISA_Generic_Builder::simd_ternary_impl(unsigned char mask, Value *a, Value *b, Value *c) {
    assert(a->getType() == b->getType());
    assert(b->getType() == c->getType());

    if (mask == 0) {
        return Constant::getNullValue(a->getType());
    }
    if (mask == 0xFF) {
        return Constant::getAllOnesValue(a->getType());
    }

    unsigned char not_a_mask = mask & 0x0F;
    unsigned char a_mask = (mask >> 4) & 0x0F;
    if (a_mask == not_a_mask) {
        return simd_binary(a_mask, b, c);
    }

    unsigned char b_mask = ((mask & 0xC0) >> 4) | ((mask & 0x0C) >> 2);
    unsigned char not_b_mask = ((mask & 0x30) >> 2) | (mask & 0x03);
    if (b_mask == not_b_mask) {
        return simd_binary(b_mask, a, c);
    }

    unsigned char c_mask = ((mask & 0x80) >> 4) | ((mask & 0x20) >> 3) | ((mask & 0x08) >> 2) | ((mask & 02) >> 1);
    unsigned char not_c_mask = ((mask & 0x40) >> 3) | ((mask & 0x10) >> 2) | ((mask & 0x04) >> 1) | (mask & 01);
    if (c_mask == not_c_mask) {
        return simd_binary(c_mask, a, b);
    }

    Value *bc_hi = simd_binary(a_mask, b, c);
    Value *bc_lo = simd_binary(not_a_mask, b, c);
    Value *a_bc = mCB->CreateAnd(a, bc_hi);
    Value *not_a_bc = mCB->CreateAnd(mCB->CreateNot(a), bc_lo);
    return mCB->CreateOr(a_bc, not_a_bc);
}

Value *IDISA_Generic_Builder::esimd_mergeh_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 8) {
        if (getVectorBitWidth(a) > mNativeBitBlockWidth) {
            Value *a_hi = CreateHalfVectorHigh(a);
            Value *b_hi = CreateHalfVectorHigh(b);
            return CreateDoubleVector(esimd_mergel(fw, a_hi, b_hi), esimd_mergeh(fw, a_hi, b_hi));
        }
        Value *abh = simd_or(simd_select_hi(fw * 2, b), simd_srli(bitManipFW, simd_select_hi(fw * 2, a), fw));
        Value *abl = simd_or(simd_slli(bitManipFW, simd_select_lo(fw * 2, b), fw), simd_select_lo(fw * 2, a));
        return esimd_mergeh(fw * 2, abl, abh);
    }
    const auto field_count = getVectorBitWidth(a) / fw;

    IndexVector Idxs(field_count);
    for (unsigned i = 0; i < field_count / 2; i++) {
        Idxs[2 * i] = mCB->getInt32(i + field_count / 2);                   // selects elements from first reg.
        Idxs[2 * i + 1] = mCB->getInt32(i + field_count / 2 + field_count); // selects elements from second reg.
    }
    return mCB->CreateShuffleVector(fwCast(fw, a), fwCast(fw, b), ConstantVector::get(Idxs));
}

Value *IDISA_Generic_Builder::esimd_mergel_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 8) {
        if (getVectorBitWidth(a) > mNativeBitBlockWidth) {
            Value *a_lo = CreateHalfVectorLow(a);
            Value *b_lo = CreateHalfVectorLow(b);
            return CreateDoubleVector(esimd_mergel(fw, a_lo, b_lo), esimd_mergeh(fw, a_lo, b_lo));
        }
        Value *abh = simd_or(simd_select_hi(fw * 2, b), simd_srli(bitManipFW, simd_select_hi(fw * 2, a), fw));
        Value *abl = simd_or(simd_slli(bitManipFW, simd_select_lo(fw * 2, b), fw), simd_select_lo(fw * 2, a));
        return esimd_mergel(fw * 2, abl, abh);
    }
    const auto field_count = getVectorBitWidth(a) / fw;
    IndexVector Idxs(field_count);
    for (unsigned i = 0; i < field_count / 2; i++) {
        Idxs[2 * i] = mCB->getInt32(i);                   // selects elements from first reg.
        Idxs[2 * i + 1] = mCB->getInt32(i + field_count); // selects elements from second reg.
    }
    return mCB->CreateShuffleVector(fwCast(fw, a), fwCast(fw, b), ConstantVector::get(Idxs));
}

Value *IDISA_Generic_Builder::esimd_bitspread_impl(unsigned vec_width, unsigned fw, Value *bitmask) {
    const size_t fieldCount = vec_width / fw;
    Type *maskVecTy = FixedVectorType::get(mCB->getInt1Ty(), fieldCount);
    Type *spreadVecTy = FixedVectorType::get(mCB->getIntNTy(fw), fieldCount);
    return mCB->CreateZExt(mCB->CreateBitCast(mCB->CreateZExtOrTrunc(bitmask, mCB->getIntNTy(fieldCount)), maskVecTy),
                           spreadVecTy);
}

Value *IDISA_Generic_Builder::hsimd_packh_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 2)
        UnsupportedFieldWidthError(fw, "hsimd_packh");
    if (fw / 2 < 8) {
        assert(bitManipFW >= fw * 2);
        Value *aLo = simd_srli(bitManipFW, a, fw / 2);
        Value *bLo = simd_srli(bitManipFW, b, fw / 2);
        return hsimd_packl(fw, aLo, bLo);
    }
    Value *aVec = fwCast(fw / 2, a);
    Value *bVec = fwCast(fw / 2, b);
    const auto field_count = 2 * getVectorBitWidth(a) / fw;
    IndexVector Idxs(field_count);
    for (unsigned i = 0; i < field_count; i++) {
        Idxs[i] = mCB->getInt32(2 * i + 1);
    }
    return mCB->CreateShuffleVector(aVec, bVec, ConstantVector::get(Idxs));
}

Value *IDISA_Generic_Builder::hsimd_packl_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 2)
        UnsupportedFieldWidthError(fw, "hsimd_packl");
    if (fw / 2 < 8) {
        assert(bitManipFW >= fw * 2);
        Value *aLo = simd_select_lo(fw, a);
        Value *bLo = simd_select_lo(fw, b);
        return hsimd_packl(fw * 2, simd_or(simd_srli(bitManipFW, aLo, fw / 2), aLo),
                           simd_or(simd_srli(bitManipFW, bLo, fw / 2), bLo));
    }
    Value *aVec = fwCast(fw / 2, a);
    Value *bVec = fwCast(fw / 2, b);
    const auto field_count = 2 * getVectorBitWidth(a) / fw;
    IndexVector Idxs(field_count);
    for (unsigned i = 0; i < field_count; i++) {
        Idxs[i] = mCB->getInt32(2 * i);
    }
    return mCB->CreateShuffleVector(aVec, bVec, ConstantVector::get(Idxs));
}

Value *IDISA_Generic_Builder::hsimd_packss_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 2)
        UnsupportedFieldWidthError(fw, "hsimd_packss");
    const unsigned vectorWidth = getVectorBitWidth(a);
    Constant *top_bit = Constant::getIntegerValue(mCB->getIntNTy(vectorWidth),
                                                  APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw / 2, 1)));
    Value *hi = hsimd_packh(fw, a, b);
    Value *lo = hsimd_packl(fw, a, b);
    Value *bits_that_must_match_sign = simd_if(1, top_bit, lo, hi);
    Value *sign_mask = simd_srai(fw / 2, hi, fw / 2 - 1);
    Value *safe = simd_eq(fw / 2, sign_mask, bits_that_must_match_sign);
    return simd_if(fw / 2, safe, lo, simd_eq(1, top_bit, sign_mask));
}

Value *IDISA_Generic_Builder::hsimd_packus_impl(unsigned fw, Value *a, Value *b) {
    if (fw < 2)
        UnsupportedFieldWidthError(fw, "hsimd_packus");
    Value *hi = hsimd_packh(fw, a, b);
    Value *lo = hsimd_packl(fw, a, b);
    Value *high_mask = simd_gt(fw / 2, hi, ConstantVector::getNullValue(a->getType()));
    Value *low_mask = simd_ge(fw / 2, hi, ConstantVector::getNullValue(a->getType()));
    return simd_and(simd_or(high_mask, lo), low_mask);
}

Value *IDISA_Generic_Builder::hsimd_packh_in_lanes_impl(unsigned lanes, unsigned fw, Value *a, Value *b) {
    if (fw < 16)
        UnsupportedFieldWidthError(fw, "packh_in_lanes");
    const unsigned fw_out = fw / 2;
    const unsigned fields_per_lane = getBitBlockWidth() / (fw_out * lanes);
    const unsigned field_offset_for_b = getBitBlockWidth() / fw_out;
    const unsigned field_count = getBitBlockWidth() / fw_out;
    IndexVector Idxs(field_count);
    for (unsigned lane = 0, j = 0; lane < lanes; lane++) {
        const unsigned first_field_in_lane = lane * fields_per_lane; // every second field
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = mCB->getInt32(first_field_in_lane + (2 * i) + 1);
        }
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = mCB->getInt32(field_offset_for_b + first_field_in_lane + (2 * i) + 1);
        }
    }
    return mCB->CreateShuffleVector(fwCast(fw_out, a), fwCast(fw_out, b), ConstantVector::get(Idxs));
}

Value *IDISA_Generic_Builder::hsimd_packl_in_lanes_impl(unsigned lanes, unsigned fw, Value *a, Value *b) {
    if (fw < 16)
        UnsupportedFieldWidthError(fw, "packl_in_lanes");
    const unsigned fw_out = fw / 2;
    const unsigned fields_per_lane = getBitBlockWidth() / (fw_out * lanes);
    const unsigned field_offset_for_b = getBitBlockWidth() / fw_out;
    const unsigned field_count = getBitBlockWidth() / fw_out;
    IndexVector Idxs(field_count);
    for (unsigned lane = 0, j = 0; lane < lanes; lane++) {
        const unsigned first_field_in_lane = lane * fields_per_lane; // every second field
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = mCB->getInt32(first_field_in_lane + (2 * i));
        }
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = mCB->getInt32(field_offset_for_b + first_field_in_lane + (2 * i));
        }
    }
    return mCB->CreateShuffleVector(fwCast(fw_out, a), fwCast(fw_out, b), ConstantVector::get(Idxs));
}

Value *IDISA_Generic_Builder::hsimd_signmask_impl(unsigned fw, Value *a) {
    if (fw < 8)
        UnsupportedFieldWidthError(fw, "hsimd_signmask");
    Value *a1 = fwCast(fw, a);
    Value *mask = mCB->CreateICmpSLT(a1, ConstantAggregateZero::get(a1->getType()));
    const auto maskWidth = getBitBlockWidth() / fw;
    mask = mCB->CreateBitCast(mask, mCB->getIntNTy(maskWidth));
    if (maskWidth < bitManipFW) {
        mask = mCB->CreateZExt(mask, mCB->getInt32Ty());
    }
    return mask;
}

Value *IDISA_Generic_Builder::mvmd_extract_impl(unsigned fw, Value *a, unsigned fieldIndex) {
    if (fw < 8) {
        unsigned byte_no = (fieldIndex * fw) / 8;
        unsigned intrabyte_shift = (fieldIndex * fw) % 8;
        Value *byte = mCB->CreateExtractElement(fwCast(8, a), mCB->getInt32(byte_no));
        return mCB->CreateTrunc(mCB->CreateLShr(byte, mCB->getInt8(intrabyte_shift)), mCB->getIntNTy(fw));
    }
    return mCB->CreateExtractElement(fwCast(fw, a), mCB->getInt32(fieldIndex));
}

Value *IDISA_Generic_Builder::mvmd_insert_impl(unsigned fw, Value *a, Value *elt, unsigned fieldIndex) {
    if (fw < 8) {
        unsigned byte_no = (fieldIndex * fw) / 8;
        unsigned intrabyte_shift = (fieldIndex * fw) % 8;
        unsigned field_mask = ((1 << fw) - 1) << intrabyte_shift;
        Value *byte = mCB->CreateAnd(mCB->CreateExtractElement(fwCast(8, a), mCB->getInt32(byte_no)),
                                     mCB->getInt8(0xFF & ~field_mask));
        byte = mCB->CreateOr(
            byte, mCB->CreateShl(mCB->CreateZExtOrTrunc(elt, mCB->getInt8Ty()), mCB->getInt8(intrabyte_shift)));
        return mCB->CreateInsertElement(fwCast(8, a), byte, mCB->getInt32(byte_no));
    }
    return mCB->CreateInsertElement(fwCast(fw, a), elt, mCB->getInt32(fieldIndex));
}

Value *IDISA_Generic_Builder::mvmd_slli_impl(unsigned fw, Value *a, unsigned shift) {
    if (shift == 0)
        return a;
    Value *r = mvmd_dslli(fw, a, allZeroes(), shift);
    return r;
}

Value *IDISA_Generic_Builder::mvmd_srli_impl(unsigned fw, Value *a, unsigned shift) {
    if (shift == 0)
        return a;
    const auto field_count = getVectorBitWidth(a) / fw;
    return mvmd_dslli(fw, allZeroes(), a, field_count - shift);
}

Value *IDISA_Generic_Builder::mvmd_dslli_impl(unsigned fw, Value *a, Value *b, unsigned shift) {
    if (shift == 0)
        return a;
    if (fw > 32) {
        return mvmd_dslli(32, a, b, shift * (fw / 32));
    } else if (((shift % 2) == 0) && (fw < 32)) {
        return mvmd_dslli(2 * fw, a, b, shift / 2);
    }
    if (fw >= 16) {
        const auto field_count = getVectorBitWidth(a) / fw;
        IndexVector Idxs(field_count);
        for (unsigned i = 0; i < field_count; i++) {
            Idxs[i] = mCB->getInt32(i + field_count - shift);
        }
        return mCB->CreateShuffleVector(fwCast(fw, b), fwCast(fw, a), ConstantVector::get(Idxs));
    } else {
        unsigned field32_shift = (shift * fw) / 32;
        unsigned bit_shift = (shift * fw) % 32;
        Value *const L = simd_slli(32, mvmd_dslli(32, a, b, field32_shift), bit_shift);
        Value *const R = simd_srli(32, mvmd_dslli(32, a, b, field32_shift + 1), 32 - bit_shift);
        return simd_or(L, R);
    }
}

//
//  Generic mvmd_shuffle reduces to byte shuffling at the native SIMD width.
Value *IDISA_Generic_Builder::mvmd_shuffle_impl(unsigned fw, Value *data_table, Value *index_vector, ShuffleMode mode) {
    auto vec_width = getVectorBitWidth(data_table);
    // llvm::errs() << "IDISA_Generic_Builder::mvmd_shuffle , vec_width = " << vec_width << ", fw = " << fw <<
    // "\n";
    if (vec_width == fw) {
        // Special case for a vector with a single field.
        if (mode == ShuffleMode::TruncateIndex) {
            return data_table;
        }
        Value *isIndex0 = mCB->CreateIsNull(index_vector);
        return mCB->CreateSelect(isIndex0, data_table, ConstantInt::getNullValue(data_table->getType()));
    }
    if (vec_width > mNativeBitBlockWidth) {
        auto fieldCount = vec_width / fw;
        if (fw >= 16) {
            Value *t0 = CreateHalfVectorLow(data_table);
            Value *t1 = CreateHalfVectorHigh(data_table);
            Value *hi_fields = hsimd_packh(fw, t0, t1);
            Value *lo_fields = hsimd_packl(fw, t0, t1);
            if (mode == ShuffleMode::ZeroOnHighIndexBit) {
                // Modify the index fields so that the low half of
                // each field is the proper index with high bit set
                // for the narrower shuffle.
                Value *shifted_idx = simd_srli(fw, index_vector, fw / 2);
                Value *high_bit_of_low_half =
                    getSplat(fieldCount, ConstantInt::get(mCB->getIntNTy(fw), 1 << (fw / 2 - 1)));
                index_vector = simd_if(1, high_bit_of_low_half, shifted_idx, index_vector);
            } else if (mode == ShuffleMode::ZeroOnIndexOver) {
                Value *over =
                    simd_uge(fw, index_vector, getSplat(fieldCount, ConstantInt::get(mCB->getIntNTy(fw), fieldCount)));
                index_vector = simd_or(over, index_vector);
            }
            Value *ix0 = CreateHalfVectorLow(index_vector);
            Value *ix1 = CreateHalfVectorHigh(index_vector);
            Value *packed_ix = hsimd_packl(fw, ix0, ix1);
            Value *shuf_lo = mvmd_shuffle(fw / 2, lo_fields, packed_ix, mode);
            Value *shuf_hi = mvmd_shuffle(fw / 2, hi_fields, packed_ix, mode);
            Value *merge0 = esimd_mergel(fw / 2, shuf_lo, shuf_hi);
            Value *merge1 = esimd_mergeh(fw / 2, shuf_lo, shuf_hi);
            return fwCast(fw, CreateDoubleVector(merge0, merge1));
        } else if (fw == 8) {
            Value *t0 = CreateHalfVectorLow(data_table);
            Value *t1 = CreateHalfVectorHigh(data_table);
            Value *ix0 = CreateHalfVectorLow(index_vector);
            Value *ix1 = CreateHalfVectorHigh(index_vector);
            Value *shuf0 = mvmd_shuffle2(fw, t0, t1, ix0, mode);
            Value *shuf1 = mvmd_shuffle2(fw, t0, t1, ix1, mode);
            return fwCast(fw, CreateDoubleVector(shuf0, shuf1));
        }
    }
    if ((vec_width == mNativeBitBlockWidth) && ((fw == 16) || (fw == 32) || (fw == 64))) {
        // Create a table for shuffling with smaller field widths.
        const unsigned fieldCount = vec_width / fw;
        Constant *fieldMask = getSplat(fieldCount, ConstantInt::get(mCB->getIntNTy(fw), fieldCount - 1));
        Value *inbounds_idx = simd_and(index_vector, fieldMask);
        ConstantInt *multiplier = 0;
        ConstantInt *addition = 0;
        if (fw == 64) {
            multiplier = mCB->getInt64(0x0808080808080808ULL);
            addition = mCB->getInt64(0x0706050403020100ULL);
        } else if (fw == 32) {
            multiplier = mCB->getInt32(0x04040404);
            addition = mCB->getInt32(0x03020100);
        } else if (fw == 16) {
            multiplier = mCB->getInt16(0x0202);
            addition = mCB->getInt16(0x0100);
        }
        Value *narrowed_idx = mCB->CreateMul(fwCast(fw, inbounds_idx), getSplat(fieldCount, multiplier));
        narrowed_idx = mCB->CreateOr(narrowed_idx, getSplat(fieldCount, addition));
        if (mode == ShuffleMode::ZeroOnHighIndexBit) {
            narrowed_idx = simd_or(narrowed_idx, simd_lt(fw, index_vector, allZeroes()));
        } else if (mode == ShuffleMode::ZeroOnIndexOver) {
            Value *out_of_bounds = simd_ugt(fw, index_vector, fieldMask);
            narrowed_idx = simd_or(out_of_bounds, narrowed_idx);
        }
        return fwCast(fw, mvmd_shuffle(8, data_table, narrowed_idx, mode));
    }
    UnsupportedFieldWidthError(fw, "mvmd_shuffle");
}

Value *IDISA_Generic_Builder::mvmd_shuffle2_impl(unsigned fw, Value *table0, Value *table1, Value *index_vector,
                                                 ShuffleMode mode) {
    auto vec_width = getVectorBitWidth(table0);
    //  Use two shuffles, with selection by the bit value within the shuffle_table.
    const auto field_count = vec_width / fw;
    Constant *selectorSplat = getSplat(field_count, ConstantInt::get(mCB->getIntNTy(fw), field_count));
    Value *selectMask = simd_eq(fw, simd_and(index_vector, selectorSplat), selectorSplat);
    Value *idx = simd_and(index_vector, simd_not(selectorSplat));
    return simd_or(simd_and(mvmd_shuffle(fw, table0, idx, mode), simd_not(selectMask)),
                   simd_and(mvmd_shuffle(fw, table1, idx, mode), selectMask));
}

Value *IDISA_Generic_Builder::mvmd_compress_impl(unsigned fw, Value *v, Value *select_mask) {
    unsigned vec_width = getVectorBitWidth(v);
    if (LLVM_UNLIKELY(fw < 8)) {
        UnsupportedFieldWidthError(fw, "mvmd_compress");
    } else {
        IntegerType *const fieldTy = mCB->getIntNTy(fw);
        const auto fieldCount = vec_width / fw;
        Type *maskTy = select_mask->getType();
        if (maskTy->isIntegerTy()) {
            if (fieldCount <= fw) {
                IndexVector elements(fieldCount);
                for (unsigned i = 0; i < fieldCount; i++) {
                    elements[i] = ConstantInt::get(fieldTy, 1ULL << i);
                }
                Constant *seq = ConstantVector::get(elements);
                select_mask = simd_eq(fw, simd_and(simd_fill(fw, select_mask), seq), seq);
            } else {
                Value *const spread_mask = esimd_bitspread(vec_width, fw, select_mask);
                select_mask = simd_any(fw, spread_mask);
            }
        }
        Value *selected = simd_and(v, select_mask);
        Constant *oneSplat = getSplat(fieldCount, ConstantInt::get(fieldTy, 1));
        Value *deletion_counts = simd_add(fw, oneSplat, select_mask);
        Value *deletion_totals = hsimd_partial_sum(fw, deletion_counts);
        unsigned shiftAmount = 1;
        while (shiftAmount < fieldCount) {
            Value *shift_splat = getSplat(fieldCount, ConstantInt::get(fieldTy, shiftAmount));
            Value *shift_select = simd_and(deletion_totals, shift_splat);
            Value *shift_mask = simd_eq(fw, shift_select, shift_splat);
            Value *to_shift = simd_and(shift_mask, selected);
            Value *shifted = mvmd_srli(fw, to_shift, shiftAmount);
            deletion_totals = simd_sub(fw, deletion_totals, shift_select);
            selected = simd_or(shifted, simd_xor(selected, to_shift));
            shiftAmount *= 2;
        }
        return selected;
    }
}

Value *IDISA_Generic_Builder::mvmd_expand_impl(unsigned fw, Value *v, Value *select_mask) {
    unsigned vec_width = getVectorBitWidth(v);
    unsigned field_count = vec_width / fw;
    Type *maskTy = select_mask->getType();
    if (maskTy->isIntegerTy()) {
        select_mask = esimd_bitspread(vec_width, fw, select_mask);
    } else {
        Constant *oneSplat = getSplat(field_count, ConstantInt::get(mCB->getIntNTy(fw), 1));
        select_mask = simd_and(select_mask, oneSplat);
    }
    Value *prior_counts = mvmd_slli(fw, hsimd_partial_sum(fw, select_mask), 1);
    Value *spread_data = mvmd_shuffle(fw, v, prior_counts);
    return mCB->CreateAnd(spread_data, simd_any(fw, select_mask));
}

Value *IDISA_Generic_Builder::bitblock_any_impl(Value *a) {
    Type *aType = a->getType();
    if (aType->isIntegerTy()) {
        return mCB->CreateICmpNE(a, ConstantInt::getNullValue(aType));
    } else {
        Value *r = simd_ne(bitManipFW, a, allZeroes());
        r = hsimd_signmask(bitManipFW, r);
        return mCB->CreateICmpNE(r, ConstantInt::getNullValue(r->getType()), "bitblock_any");
    }
}

// full add producing {carryout, sum}
std::pair<Value *, Value *> IDISA_Generic_Builder::bitblock_add_with_carry_impl(Value *a, Value *b, Value *carryin) {
    Type *const carryTy = carryin->getType();
    if (carryTy != getBitBlockType()) {
        assert(carryTy->isIntegerTy());
        if (LLVM_LIKELY(carryTy->getIntegerBitWidth() < getLaneWidth())) {
            carryin = mCB->CreateZExt(carryin, getLaneTy());
        }
        carryin = mCB->CreateInsertElement(ConstantVector::getNullValue(a->getType()), carryin, mCB->getInt32(0));
    }
    Value *carrygen = simd_and(a, b);
    Value *carryprop = simd_or(a, b);
    Value *sum = simd_add(getBitBlockWidth(), simd_add(getBitBlockWidth(), a, b), carryin);
    Value *carryout = simd_or(carrygen, simd_and(carryprop, mCB->CreateNot(sum)));
    carryout = simd_srli(getBitBlockWidth(), carryout, getBitBlockWidth() - 1);
    carryout = mCB->CreateBitCast(carryout, getBitBlockType());
    if (carryout->getType() != carryTy) {
        carryout = mCB->CreateExtractElement(carryout, mCB->getInt32(0));
        carryout = mCB->CreateZExtOrTrunc(carryout, carryTy);
    }
    return std::pair<Value *, Value *>(carryout, bitCast(sum));
}

// full subtract producing {borrowOut, difference}
std::pair<Value *, Value *> IDISA_Generic_Builder::bitblock_subtract_with_borrow_impl(Value *a, Value *b,
                                                                                      Value *borrowIn) {
    Value *in = borrowIn;
    Type *borrowInTy = borrowIn->getType();
    if (borrowInTy != getBitBlockType()) {
        assert(borrowInTy->isIntegerTy());
        in = mCB->CreateZExtOrTrunc(in, getBitBlockType()->getElementType());
        in = mCB->CreateInsertElement(Constant::getNullValue(getBitBlockType()), in, mCB->getInt32(0));
    }
    Value *partial = simd_sub(getBitBlockWidth(), simd_sub(getBitBlockWidth(), a, b), in);
    Value *borrowOut = simd_srli(getBitBlockWidth(), partial, getBitBlockWidth() - 1);

    borrowOut = mCB->CreateBitCast(borrowOut, getBitBlockType());
    if (borrowInTy != getBitBlockType()) {
        borrowOut = mCB->CreateExtractElement(borrowOut, mCB->getInt32(0));
        borrowOut = mCB->CreateZExtOrTrunc(borrowOut, borrowInTy);
    }
    return std::make_pair(borrowOut, bitCast(partial));
}

// full shift producing {shiftout, shifted}
std::pair<Value *, Value *> IDISA_Generic_Builder::bitblock_advance_impl(Value *const a, Value *shiftin,
                                                                         const unsigned shift) {
    Value *shifted = nullptr;
    Value *shiftout = nullptr;
    Type *shiftTy = shiftin->getType();
    assert(a->getType() == getBitBlockType());
    assert(getBitBlockType()->getElementType()->getIntegerBitWidth() == getLaneWidth());
    if (LLVM_UNLIKELY(shift == 0)) {
        return std::pair<Value *, Value *>(Constant::getNullValue(shiftTy), a);
    }
    if (shiftTy != getBitBlockType()) {
        assert(shiftTy->isIntegerTy());
        if (LLVM_LIKELY(shiftTy->getIntegerBitWidth() < getLaneWidth())) {
            shiftin = mCB->CreateZExt(shiftin, getLaneTy());
        }
        shiftin = mCB->CreateInsertElement(ConstantVector::getNullValue(a->getType()), shiftin, mCB->getInt32(0));
    }
    assert(shiftin->getType() == getBitBlockType());

    auto getShiftout = [&](Value *v) {
        if (v->getType() != shiftTy) {
            v = mCB->CreateExtractElement(v, mCB->getInt32(0));
            if (LLVM_LIKELY(shiftTy->getIntegerBitWidth() < getLaneWidth())) {
                v = mCB->CreateTrunc(v, shiftTy);
            }
        }
        return v;
    };

    if (LLVM_UNLIKELY(shift == getBitBlockWidth())) {
        return std::pair<Value *, Value *>(getShiftout(a), shiftin);
    }
#ifndef LEAVE_CARRY_UNNORMALIZED
    if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
        shifted = mCB->CreateOr(bitCast(mvmd_slli(8, a, shift / 8)), shiftin);
        shiftout = bitCast(mvmd_srli(8, a, (getBitBlockWidth() - shift) / 8));
    } else {
        Value *shiftback = simd_srli(getLaneWidth(), a, getLaneWidth() - (shift % getLaneWidth()));
        Value *shiftfwd = simd_slli(getLaneWidth(), a, shift % getLaneWidth());
        if (LLVM_LIKELY(shift < getLaneWidth())) {
            shiftout = mvmd_srli(getLaneWidth(), shiftback, getBitBlockWidth() / getLaneWidth() - 1);
            shifted = mCB->CreateOr(mCB->CreateOr(shiftfwd, shiftin), mvmd_slli(getLaneWidth(), shiftback, 1));
        } else {
            shiftout = mCB->CreateOr(shiftback, mvmd_srli(getLaneWidth(), shiftfwd, 1));
            shifted = mCB->CreateOr(shiftin,
                                    mvmd_slli(getLaneWidth(), shiftfwd, (getBitBlockWidth() - shift) / getLaneWidth()));
            if ((shift + getLaneWidth()) < getBitBlockWidth()) {
                shiftout = mvmd_srli(getLaneWidth(), shiftout, (getBitBlockWidth() - shift) / getLaneWidth());
                shifted = mCB->CreateOr(shifted, mvmd_slli(getLaneWidth(), shiftback, shift / getLaneWidth() + 1));
            }
        }
    }
#else
    shiftout = a;
    if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
        shifted = mvmd_dslli(8, a, shiftin, (getBitBlockWidth() - shift) / 8);
    } else if (LLVM_LIKELY(shift < getLaneWidth())) {
        Value *ahead = mvmd_dslli(getLaneWidth(), a, shiftin, getBitBlockWidth() / getLaneWidth() - 1);
        shifted =
            simd_or(simd_srli(getLaneWidth(), ahead, getLaneWidth() - shift), simd_slli(getLaneWidth(), a, shift));
    } else {
        throw std::runtime_error("Unsupported shift.");
    }
#endif
    assert(shifted->getType() == getBitBlockType());
    assert(shiftout->getType() == getBitBlockType());
    return std::pair<Value *, Value *>(getShiftout(shiftout), shifted);
}

// full shift producing {shiftout, shifted}
std::pair<Value *, Value *> IDISA_Generic_Builder::bitblock_indexed_advance_impl(Value *strm, Value *index_strm,
                                                                                 Value *shiftIn, unsigned shiftAmount) {
    const unsigned bitWidth = mCB->getSizeTy()->getBitWidth();
    Type *const iBitBlock = mCB->getIntNTy(getBitBlockWidth());
    Value *const shiftVal = mCB->getSize(shiftAmount);
    Value *const extracted_bits = simd_pext(bitWidth, strm, index_strm);
    Value *const ix_popcounts = simd_popcount(bitWidth, index_strm);
    const auto n = getBitBlockWidth() / bitWidth;
    FixedVectorType *const vecTy = FixedVectorType::get(mCB->getSizeTy(), n);

    Value *carryOut = nullptr;
    Value *result = UndefValue::get(vecTy);
    if (LLVM_LIKELY(shiftAmount < bitWidth)) {
        Value *carry = mvmd_extract(bitWidth, shiftIn, 0);
        for (unsigned i = 0; i < n; i++) {
            Value *ix_popcnt = mvmd_extract(bitWidth, ix_popcounts, i);
            Value *bits = mvmd_extract(bitWidth, extracted_bits, i);
            Value *adv = mCB->CreateOr(mCB->CreateShl(bits, shiftVal), carry);
            // We have two cases depending on whether the popcount of the index pack is < shiftAmount or not.
            Value *popcount_small = mCB->CreateICmpULT(ix_popcnt, shiftVal);
            Value *carry_if_popcount_small = mCB->CreateOr(mCB->CreateShl(bits, mCB->CreateSub(shiftVal, ix_popcnt)),
                                                           mCB->CreateLShr(carry, ix_popcnt));
            Value *carry_if_popcount_large = mCB->CreateLShr(bits, mCB->CreateSub(ix_popcnt, shiftVal));
            carry = mCB->CreateSelect(popcount_small, carry_if_popcount_small, carry_if_popcount_large);
            result = mvmd_insert(bitWidth, result, adv, i);
        }
        carryOut = mvmd_insert(bitWidth, allZeroes(), carry, 0);
    } else if (shiftAmount <= getBitBlockWidth()) {
        // The shift amount is always greater than the popcount of the individual
        // elements that we deal with.   This simplifies some of the logic.
        Value *carry = mCB->CreateBitCast(shiftIn, iBitBlock);
        for (unsigned i = 0; i < n; i++) {
            Value *ix_popcnt = mvmd_extract(bitWidth, ix_popcounts, i);
            Value *bits =
                mvmd_extract(bitWidth, extracted_bits, i); // All these bits are shifted out (appended to carry).
            result = mvmd_insert(bitWidth, result, mvmd_extract(bitWidth, carry, 0), i);
            carry = mCB->CreateLShr(
                carry,
                mCB->CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed, make room for new bits.
            carry =
                mCB->CreateOr(carry, mCB->CreateShl(mCB->CreateZExt(bits, iBitBlock),
                                                    mCB->CreateZExt(mCB->CreateSub(shiftVal, ix_popcnt), iBitBlock)));
        }
        carryOut = carry;
    } else {
        // The shift amount is greater than the total popcount.   We will consume popcount
        // bits from the shiftIn value only, and produce a carry out value of the selected bits.
        Value *carry = mCB->CreateBitCast(shiftIn, iBitBlock);
        carryOut = ConstantInt::getNullValue(iBitBlock);
        Value *generated = mCB->getSize(0);
        for (unsigned i = 0; i < n; i++) {
            Value *ix_popcnt = mvmd_extract(bitWidth, ix_popcounts, i);
            Value *bits =
                mvmd_extract(bitWidth, extracted_bits, i); // All these bits are shifted out (appended to carry).
            result = mvmd_insert(bitWidth, result, mvmd_extract(bitWidth, carry, 0), i);
            carry = mCB->CreateLShr(carry, mCB->CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed.
            carryOut = mCB->CreateOr(
                carryOut, mCB->CreateShl(mCB->CreateZExt(bits, iBitBlock), mCB->CreateZExt(generated, iBitBlock)));
            generated = mCB->CreateAdd(generated, ix_popcnt);
        }
    }
    return std::pair<Value *, Value *>{bitCast(carryOut), simd_pdep(bitWidth, result, index_strm)};
}

Value *IDISA_Generic_Builder::bitblock_mask_from_impl(Value *const position, const bool safe) {
    // We use fields of size bitManipFW to do math on bit indices spanning the whole SIMD block
    assert(bitManipFW > floor_log2(2 * getBitBlockWidth() - 1));

    Value *const originalPos = mCB->CreateZExtOrTrunc(position, mCB->getIntNTy(bitManipFW));
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant *const BLOCK_WIDTH = mCB->getIntN(bitManipFW, getBitBlockWidth());
        mCB->CreateAssert(mCB->CreateICmpULT(originalPos, BLOCK_WIDTH), "position exceeds block width");
    }
    Value *const pos = safe ? position : mCB->CreateAnd(originalPos, mCB->getIntN(bitManipFW, getBitBlockWidth() - 1));
    const auto fieldCount = getBitBlockWidth() / bitManipFW;
    IndexVector posBase(fieldCount);
    for (unsigned i = 0; i < fieldCount; i++) {
        posBase[i] = mCB->getIntN(bitManipFW, bitManipFW * i);
    }
    Value *const posBaseVec = ConstantVector::get(posBase);
    Value *const positionVec = simd_fill(bitManipFW, pos);
    Value *const fullFieldWidthMasks =
        mCB->CreateSExt(mCB->CreateICmpUGT(posBaseVec, positionVec), fwVectorType(bitManipFW));
    Constant *const FIELD_ONES = ConstantInt::getAllOnesValue(mCB->getIntNTy(bitManipFW));
    Value *const bitField = mCB->CreateShl(FIELD_ONES, mCB->CreateAnd(pos, mCB->getIntN(bitManipFW, bitManipFW - 1)));
    Value *const fieldNo = mCB->CreateLShr(pos, mCB->getIntN(bitManipFW, floor_log2(bitManipFW)));
    Value *result = mCB->CreateInsertElement(fullFieldWidthMasks, bitField, fieldNo);
    if (!safe) { // if the originalPos doesn't match the moddedPos then the originalPos must exceed the block width.
        Constant *const VECTOR_ZEROES = Constant::getNullValue(fwVectorType(bitManipFW));
        result = mCB->CreateSelect(mCB->CreateICmpEQ(originalPos, pos), result, VECTOR_ZEROES);
    }
    return bitCast(result);
}

Value *IDISA_Generic_Builder::bitblock_mask_to_impl(Value *const position, const bool safe) {
    // We use fields of size bitManipFW to do math on bit indices spanning the whole SIMD block
    assert(bitManipFW > floor_log2(2 * getBitBlockWidth() - 1));

    Value *const originalPos = mCB->CreateZExtOrTrunc(position, mCB->getIntNTy(bitManipFW));
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant *const BLOCK_WIDTH = mCB->getIntN(bitManipFW, getBitBlockWidth());
        mCB->CreateAssert(mCB->CreateICmpULT(originalPos, BLOCK_WIDTH), "position exceeds block width");
    }
    Value *const pos = safe ? position : mCB->CreateAnd(originalPos, mCB->getIntN(bitManipFW, getBitBlockWidth() - 1));
    const auto fieldCount = getBitBlockWidth() / bitManipFW;
    IndexVector posBase(fieldCount);
    for (unsigned i = 0; i < fieldCount; i++) {
        posBase[i] = mCB->getIntN(bitManipFW, bitManipFW * i);
    }
    Value *const posBaseVec = ConstantVector::get(posBase);
    Value *const positionVec = simd_fill(bitManipFW, pos);
    Value *const fullFieldWidthMasks =
        mCB->CreateSExt(mCB->CreateICmpULT(posBaseVec, positionVec), fwVectorType(bitManipFW));
    Constant *const FIELD_ONES = ConstantInt::getAllOnesValue(mCB->getIntNTy(bitManipFW));
    Value *const bitField =
        mCB->CreateLShr(FIELD_ONES, mCB->CreateAnd(mCB->getIntN(bitManipFW, bitManipFW - 1), mCB->CreateNot(pos)));
    Value *const fieldNo = mCB->CreateLShr(pos, mCB->getIntN(bitManipFW, floor_log2(bitManipFW)));
    Value *result = mCB->CreateInsertElement(fullFieldWidthMasks, bitField, fieldNo);
    if (!safe) { // if the originalPos doesn't match the moddedPos then the originalPos must exceed the block width.
        Constant *const VECTOR_ONES = Constant::getAllOnesValue(fwVectorType(bitManipFW));
        result = mCB->CreateSelect(mCB->CreateICmpEQ(originalPos, pos), result, VECTOR_ONES);
    }
    return bitCast(result);
}

Value *IDISA_Generic_Builder::bitblock_set_bit_impl(Value *const position, const bool safe) {
    // We use fields of size bitManipFW to do math on bit indices spanning the whole SIMD block
    assert(bitManipFW > floor_log2(2 * getBitBlockWidth() - 1));

    Value *const originalPos = mCB->CreateZExtOrTrunc(position, mCB->getIntNTy(bitManipFW));
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant *const BLOCK_WIDTH = mCB->getIntN(bitManipFW, getBitBlockWidth());
        mCB->CreateAssert(mCB->CreateICmpULT(originalPos, BLOCK_WIDTH), "position exceeds block width");
    }
    Value *const bitField = mCB->CreateShl(mCB->getIntN(bitManipFW, 1),
                                           mCB->CreateAnd(originalPos, mCB->getIntN(bitManipFW, bitManipFW - 1)));
    Value *const pos = safe ? position : mCB->CreateAnd(originalPos, mCB->getIntN(bitManipFW, getBitBlockWidth() - 1));
    Value *const fieldNo = mCB->CreateLShr(pos, mCB->getIntN(bitManipFW, floor_log2(bitManipFW)));
    Constant *const VECTOR_ZEROES = Constant::getNullValue(fwVectorType(bitManipFW));
    Value *result = mCB->CreateInsertElement(VECTOR_ZEROES, bitField, fieldNo);
    if (!safe) { // If the originalPos doesn't match the moddedPos then the originalPos must exceed the block width.
        result = mCB->CreateSelect(mCB->CreateICmpEQ(originalPos, pos), result, VECTOR_ZEROES);
    }
    return bitCast(result);
}

} // namespace IDISA
