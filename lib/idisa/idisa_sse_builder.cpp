/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_sse_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/Module.h>
#include <sstream>

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(20, 0, 0)
#define getOrInsertDeclaration getDeclaration
#endif

using namespace llvm;

namespace IDISA {

DEFINE_BUILDER_CACHE_NAME(IDISA_SSE_Builder, "SSE", SSE_width)
DEFINE_BUILDER_CACHE_NAME(IDISA_SSE2_Builder, "SSE2", SSE_width)
DEFINE_BUILDER_CACHE_NAME(IDISA_SSSE3_Builder, "SSSE3", SSE_width)

Value *IDISA_SSE_Builder::hsimd_signmask_impl(const unsigned fw, Value *a) {
    // Produces wrong result on AVX2 with fw = 16
    // const unsigned SSE_blocks = getVectorBitWidth(a)/SSE_width;
    // if (SSE_blocks > 1) {
    //     Value * a_lo = CreateHalfVectorLow(a);
    //     Value * a_hi = CreateHalfVectorHigh(a);
    //     if ((fw == 8 * SSE_blocks) || (fw >= 32 * SSE_blocks)) {
    //         return IDISA_SSE_Builder::hsimd_signmask_impl(fw/2, IDISA_SSE_Builder::hsimd_packh_impl(fw, a_hi, a_lo));
    //     }
    //     unsigned maskWidth = getVectorBitWidth(a)/fw;
    //     Type * maskTy = getIntNTy(maskWidth);
    //     Value * mask_lo = CreateZExtOrTrunc(hsimd_signmask(fw, a_lo), maskTy);
    //     Value * mask_hi = CreateZExtOrTrunc(hsimd_signmask(fw, a_hi), maskTy);
    //     return fwCast(fw, CreateOr(CreateShl(mask_hi, maskWidth/2), mask_lo));
    // }
    // SSE special cases using Intrinsic::x86_sse_movmsk_ps (fw=32 only)
    if ((getVectorBitWidth(a) == SSE_width) && (fw == 32)) {
        Function *signmask_f32func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_sse_movmsk_ps);
        Type *bitBlock_f32type = FixedVectorType::get(mCB->getFloatTy(), SSE_width / 32);
        Value *a_as_ps = mCB->CreateBitCast(a, bitBlock_f32type);
        return fwCast(fw, mCB->CreateCall(signmask_f32func->getFunctionType(), signmask_f32func, a_as_ps));
    }
    // Otherwise use default logic.
    return IDISA_Generic_Builder::hsimd_signmask_impl(fw, a);
}

Value *IDISA_SSE_Builder::mvmd_compress_impl(unsigned fw, Value *a, Value *selector) {
    if (LLVM_LIKELY(getVectorBitWidth(a) == SSE_width)) {
        if (fw == 64) {
            Constant *keep[2] = {mCB->getInt64(1), mCB->getInt64(3)};
            Constant *keep_mask = ConstantVector::get({keep, 2});
            Constant *shift[2] = {mCB->getInt64(2), mCB->getInt64(0)};
            Constant *shifted_mask = ConstantVector::get({shift, 2});
            Value *a_srli1 = mvmd_srli(64, a, 1);
            Value *bdcst = simd_fill(64, mCB->CreateZExt(selector, mCB->getInt64Ty()));
            Value *kept = simd_and(simd_eq(64, simd_and(keep_mask, bdcst), keep_mask), a);
            Value *shifted = simd_and(a_srli1, simd_eq(64, shifted_mask, bdcst));
            return fwCast(fw, simd_or(kept, shifted));
        } else if (fw == 32) {
            Value *bdcst = simd_fill(32, mCB->CreateZExtOrTrunc(selector, mCB->getInt32Ty()));
            Constant *fieldBit[4] = {mCB->getInt32(1), mCB->getInt32(2), mCB->getInt32(4), mCB->getInt32(8)};
            Constant *fieldMask = ConstantVector::get({fieldBit, 4});
            Value *a_selected = simd_and(simd_eq(32, fieldMask, simd_and(fieldMask, bdcst)), a);
            Constant *rotateInwards[4] = {mCB->getInt32(1), mCB->getInt32(0), mCB->getInt32(3), mCB->getInt32(2)};
            Constant *rotateVector = ConstantVector::get({rotateInwards, 4});
            Value *rotated =
                mCB->CreateShuffleVector(fwCast(32, a_selected), UndefValue::get(fwVectorType(fw)), rotateVector);
            Constant *rotate_bit[2] = {mCB->getInt64(2), mCB->getInt64(4)};
            Constant *rotate_mask = ConstantVector::get({rotate_bit, 2});
            Value *rotateControl = simd_eq(64, fwCast(64, simd_and(bdcst, rotate_mask)), allZeroes());
            Value *centralResult = simd_if(1, rotateControl, rotated, a_selected);
            Value *delete_marks_lo =
                mCB->CreateAnd(mCB->CreateZExtOrTrunc(mCB->CreateNot(selector), mCB->getInt32Ty()), mCB->getInt32(3));
            Value *delCount_lo = mCB->CreateSub(delete_marks_lo, mCB->CreateLShr(delete_marks_lo, 1));
            return mvmd_srl(32, centralResult, delCount_lo);
        }
    }
    return IDISA_Generic_Builder::mvmd_compress_impl(fw, a, selector);
}

Value *IDISA_SSE2_Builder::hsimd_packh_impl(unsigned fw, Value *a, Value *b) {
    if ((fw == 16) && (getVectorBitWidth(a) == SSE_width)) {
        Function *packuswb_func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_sse2_packuswb_128);
        return fwCast(fw, mCB->CreateCall(packuswb_func->getFunctionType(), packuswb_func,
                                          {simd_srli(16, a, 8), simd_srli(16, b, 8)}));
    }
    // Otherwise use default logic.
    return IDISA_SSE_Builder::hsimd_packh_impl(fw, a, b);
}

Value *IDISA_SSE2_Builder::hsimd_packl_impl(unsigned fw, Value *a, Value *b) {
    if ((fw == 16) && (getVectorBitWidth(a) == SSE_width)) {
        Value *mask = simd_lomask(16);
        return hsimd_packus(fw, fwCast(16, simd_and(a, mask)), fwCast(16, simd_and(b, mask)));
    }
    // Otherwise use default logic.
    return IDISA_SSE_Builder::hsimd_packl_impl(fw, a, b);
}

Value *IDISA_SSE2_Builder::hsimd_packus_impl(unsigned fw, Value *a, Value *b) {
    if ((fw == 16) && (getVectorBitWidth(a) == SSE_width)) {
        Function *packuswb_func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_sse2_packuswb_128);
        return fwCast(fw,
                      mCB->CreateCall(packuswb_func->getFunctionType(), packuswb_func, {fwCast(16, a), fwCast(16, b)}));
    }
    // Otherwise use default logic.
    return IDISA_SSE_Builder::hsimd_packus_impl(fw, a, b);
}

Value *IDISA_SSE2_Builder::hsimd_signmask_impl(unsigned fw, Value *a) {
    // SSE2 special case using Intrinsic::x86_sse2_movmsk_pd (fw=32 only)
    if (getVectorBitWidth(a) == SSE_width) {
        if (fw == 64) {
            Function *signmask_f64func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_sse2_movmsk_pd);
            Type *bitBlock_f64type = FixedVectorType::get(mCB->getDoubleTy(), SSE_width / 64);
            Value *a_as_pd = mCB->CreateBitCast(a, bitBlock_f64type);
            return fwCast(fw, mCB->CreateCall(signmask_f64func->getFunctionType(), signmask_f64func, a_as_pd));
        }
        if (fw == 8) {
            Function *pmovmskb_func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_sse2_pmovmskb_128);
            return fwCast(fw, mCB->CreateCall(pmovmskb_func->getFunctionType(), pmovmskb_func, fwCast(8, a)));
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE_Builder::hsimd_signmask_impl(fw, a);
}

Value *IDISA_SSE2_Builder::mvmd_shuffle_impl(unsigned fw, Value *a, Value *index_vector, ShuffleMode mode) {
    if ((getVectorBitWidth(a) == SSE_width) && (fw == 64) && (mode == ShuffleMode::TruncateIndex)) {
        // First create a vector with exchanged values of the 2 fields.
        Constant *idx[2] = {mCB->getInt32(1), mCB->getInt32(0)};
        Value *exchanged =
            mCB->CreateShuffleVector(a, UndefValue::get(fwVectorType(fw)), ConstantVector::get({idx, 2}));
        // bits that change if the value in a needs to be exchanged.
        Value *changed = simd_xor(a, exchanged);
        // Now create a mask to select between original and exchanged values.
        Constant *xchg[2] = {mCB->getInt64(1), mCB->getInt64(0)};
        Value *xchg_vec = ConstantVector::get({xchg, 2});
        Constant *oneSplat = getSplat(2, mCB->getInt64(1));
        Value *exchange_mask = simd_eq(fw, simd_and(index_vector, oneSplat), xchg_vec);
        Value *rslt = simd_xor(simd_and(changed, exchange_mask), a);
        return fwCast(fw, rslt);
    }
    return IDISA_SSE_Builder::mvmd_shuffle_impl(fw, a, index_vector, mode);
}

std::vector<Value *> IDISA_SSE2_Builder::simd_pext_impl(unsigned fw, std::vector<Value *> v, Value *extract_mask) {
    if ((getVectorBitWidth(v[0]) == SSE_width) && (fw > 8)) {
        std::vector<Value *> w = v;
        w.push_back(extract_mask); // Compress the masks as well.
        w = simd_pext(fw / 2, w, extract_mask);
        Value *compressed_masks = simd_select_lo(fw, w.back());
        Value *multiplier = simd_add(fw, compressed_masks, simd_fill(fw, mCB->getIntN(fw, 1)));
        std::vector<Value *> c(v.size());
        for (unsigned i = 0; i < v.size(); i++) {
            c[i] =
                fwCast(fw, simd_or(simd_mult(fw, multiplier, simd_srli(fw, w[i], fw / 2)), simd_select_lo(fw, w[i])));
        }
        return c;
    }
    return IDISA_SSE_Builder::simd_pext_impl(fw, v, extract_mask);
}

Value *IDISA_SSSE3_Builder::esimd_mergeh_impl(unsigned fw, Value *a, Value *b) {
    if ((getVectorBitWidth(a) == SSE_width) && ((fw == 1) || (fw == 2))) {
        Constant *interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value *byte_merge = esimd_mergeh(8, a, b);
        Value *low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
        Value *high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
        // For each 16-bit field, interleave the low bits of the two bytes.
        low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8 - fw));
        // For each 16-bit field, interleave the high bits of the two bytes.
        high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8 - fw));
        return fwCast(fw, simd_or(low_bits, high_bits));
    }
    // Otherwise use default SSE logic.
    return IDISA_SSE2_Builder::esimd_mergeh_impl(fw, a, b);
}

Value *IDISA_SSSE3_Builder::esimd_mergel_impl(unsigned fw, Value *a, Value *b) {
    if ((getVectorBitWidth(a) == SSE_width) && ((fw == 1) || (fw == 2))) {
        Constant *interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value *byte_merge = esimd_mergel(8, a, b);
        Value *low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
        Value *high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
        // For each 16-bit field, interleave the low bits of the two bytes.
        low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8 - fw));
        // For each 16-bit field, interleave the high bits of the two bytes.
        high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8 - fw));
        return fwCast(fw, simd_or(low_bits, high_bits));
    }
    // Otherwise use default SSE2 logic.
    return IDISA_SSE2_Builder::esimd_mergel_impl(fw, a, b);
}

Value *IDISA_SSSE3_Builder::mvmd_shuffle_impl(unsigned fw, Value *data_table, Value *index_vector, ShuffleMode mode) {
    auto vec_width = getVectorBitWidth(data_table);
    if ((vec_width == SSE_width) && (fw == 8)) {
        auto fieldCount = vec_width / fw;
        Function *shuf8Func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_ssse3_pshuf_b_128);
        // Default for SSSE3 is ShuffleMode::ZeroOnHighBit
        if (mode == ShuffleMode::TruncateIndex) {
            Constant *fieldMask = mCB->getIntN(fw, fieldCount - 1);
            index_vector = simd_and(index_vector, getSplat(fieldCount, fieldMask));
        } else if (mode == ShuffleMode::ZeroOnIndexOver) {
            Constant *fieldMask = mCB->getIntN(fw, fieldCount - 1);
            index_vector = simd_and(index_vector, simd_ugt(fw, index_vector, getSplat(fieldCount, fieldMask)));
        }
        return fwCast(fw, mCB->CreateCall(shuf8Func->getFunctionType(), shuf8Func,
                                          {fwCast(8, data_table), fwCast(8, index_vector)}));
    }
    return IDISA_SSE2_Builder::mvmd_shuffle_impl(fw, data_table, index_vector, mode);
}

Value *IDISA_SSSE3_Builder::mvmd_compress_impl(unsigned fw, Value *a, Value *select_mask) {
    // if (getVectorBitWidth(a) == SSE_width) {
    //  simd_pext

    //}
    return IDISA_SSE2_Builder::mvmd_compress_impl(fw, a, select_mask);
}

Value *IDISA_SSSE3_Builder::mvmd_expand_impl(unsigned fw, Value *a, Value *select_mask) {
    // if (getVectorBitWidth(a) == SSE_width) {
    //  simd_pdep
    //}
    return IDISA_SSE2_Builder::mvmd_expand_impl(fw, a, select_mask);
}

} // namespace IDISA
