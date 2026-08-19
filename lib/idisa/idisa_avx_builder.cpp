/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <boost/intrusive/detail/math.hpp>
#include <idisa/idisa_avx_builder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <toolchain/toolchain.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
using boost::intrusive::detail::floor_log2;

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(20, 0, 0)
#define getOrInsertDeclaration getDeclaration
#endif

#define ADD_IF_FOUND(Flag, Value)                                                                                      \
    if (features.lookup(Value))                                                                                        \
    featureSet.set((size_t)codegen::Feature::Flag)

using namespace llvm;

namespace IDISA {

DEFINE_BUILDER_CACHE_NAME(IDISA_AVX_Builder, "AVX", AVX_width)

Value *IDISA_AVX_Builder::hsimd_signmask_impl(unsigned fw, Value *a) {
    // AVX2 special cases
    if (getVectorBitWidth(a) == AVX_width) {
        if (fw == 64) {
            Function *signmask_f64func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx_movmsk_pd_256);
            Type *bitBlock_f64type = FixedVectorType::get(mCB->getDoubleTy(), AVX_width / 64);
            Value *a_as_pd = mCB->CreateBitCast(a, bitBlock_f64type);
            return mCB->CreateCall(signmask_f64func->getFunctionType(), signmask_f64func, a_as_pd);
        } else if (fw == 32) {
            Function *signmask_f32func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx_movmsk_ps_256);
            Type *bitBlock_f32type = FixedVectorType::get(mCB->getFloatTy(), AVX_width / 32);
            Value *a_as_ps = mCB->CreateBitCast(a, bitBlock_f32type);
            return mCB->CreateCall(signmask_f32func->getFunctionType(), signmask_f32func, a_as_ps);
        }
    }
    // Otherwise use default SSE2 logic.
    return IDISA_SSE2_Builder::hsimd_signmask_impl(fw, a);
}

DEFINE_BUILDER_CACHE_NAME(IDISA_AVX2_Builder, "AVX2", AVX_width)

Value *IDISA_AVX2_Builder::hsimd_packh_impl(unsigned fw, Value *a, Value *b) {
    if (getVectorBitWidth(a) == AVX_width) {
        if ((fw > 8) && (fw <= 64)) {
            Value *aVec = fwCast(fw / 2, a);
            Value *bVec = fwCast(fw / 2, b);
            const auto field_count = 2 * AVX_width / fw;
            SmallVector<Constant *, 32> Idxs(field_count);
            const auto H = (field_count / 2);
            const auto Q = (field_count / 4);
            for (unsigned i = 0; i < Q; i++) {
                Idxs[i] = mCB->getInt32(2 * i);
                Idxs[i + Q] = mCB->getInt32((2 * i) + 1);
                Idxs[i + H] = mCB->getInt32((2 * i) + H);
                Idxs[i + H + Q] = mCB->getInt32((2 * i) + 1 + H);
            }
            Constant *const IdxVec = ConstantVector::get(Idxs);
            Value *shufa = mCB->CreateShuffleVector(aVec, aVec, IdxVec);
            Value *shufb = mCB->CreateShuffleVector(bVec, bVec, IdxVec);
            return hsimd_packh(AVX_width / 2, shufa, shufb);
        }
    }
    return IDISA_AVX_Builder::hsimd_packh_impl(fw, a, b);
}

Value *IDISA_AVX2_Builder::hsimd_packl_impl(unsigned fw, Value *a, Value *b) {
    if (getVectorBitWidth(a) == AVX_width) {
        if ((fw > 8) && (fw <= 64)) {
            Value *aVec = fwCast(fw / 2, a);
            Value *bVec = fwCast(fw / 2, b);
            const auto field_count = 2 * AVX_width / fw;
            SmallVector<Constant *, 16> Idxs(field_count);
            const auto H = (field_count / 2);
            const auto Q = (field_count / 4);
            for (unsigned i = 0; i < Q; i++) {
                Idxs[i] = mCB->getInt32(2 * i);
                Idxs[i + Q] = mCB->getInt32((2 * i) + 1);
                Idxs[i + H] = mCB->getInt32((2 * i) + H);
                Idxs[i + H + Q] = mCB->getInt32((2 * i) + H + 1);
            }
            Constant *const IdxVec = ConstantVector::get(Idxs);
            Value *shufa = mCB->CreateShuffleVector(aVec, aVec, IdxVec);
            Value *shufb = mCB->CreateShuffleVector(bVec, bVec, IdxVec);
            return hsimd_packl(AVX_width / 2, shufa, shufb);
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::hsimd_packl_impl(fw, a, b);
}

Value *IDISA_AVX2_Builder::esimd_mergeh_impl(unsigned fw, Value *a, Value *b) {
    if (getVectorBitWidth(a) == AVX_width) {
        if ((fw == 1) || (fw == 2)) {
            // Bit interleave using shuffle.
            Function *shufFn = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_pshuf_b);
            // Make a shuffle table that translates the lower 4 bits of each byte in
            // order to spread out the bits: xxxxdcba => .d.c.b.a
            // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
            Constant *interleave_table = bit_interleave_byteshuffle_table(fw);
            // Merge the bytes.
            Value *byte_merge = esimd_mergeh(8, a, b);
            Value *low_bits = mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                              {interleave_table, fwCast(8, simd_select_lo(8, byte_merge))});
            Value *high_bits = simd_slli(16,
                                         mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                                         {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}),
                                         fw);
            // For each 16-bit field, interleave the low bits of the two bytes.
            low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8 - fw));
            // For each 16-bit field, interleave the high bits of the two bytes.
            high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8 - fw));
            return simd_or(low_bits, high_bits);
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::esimd_mergeh_impl(fw, a, b);
}

Value *IDISA_AVX2_Builder::esimd_mergel_impl(unsigned fw, Value *a, Value *b) {
    if (getVectorBitWidth(a) == AVX_width) {
        if ((fw == 1) || (fw == 2)) {
            // Bit interleave using shuffle.
            Function *shufFn = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_pshuf_b);
            // Make a shuffle table that translates the lower 4 bits of each byte in
            // order to spread out the bits: xxxxdcba => .d.c.b.a
            // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
            Constant *interleave_table = bit_interleave_byteshuffle_table(fw);
            // Merge the bytes.
            Value *byte_merge = esimd_mergel(8, a, b);
            Value *low_bits = mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                              {interleave_table, fwCast(8, simd_select_lo(8, byte_merge))});
            Value *high_bits = simd_slli(16,
                                         mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                                         {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}),
                                         fw);
            // For each 16-bit field, interleave the low bits of the two bytes.
            low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8 - fw));
            // For each 16-bit field, interleave the high bits of the two bytes.
            high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8 - fw));
            return simd_or(low_bits, high_bits);
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::esimd_mergel_impl(fw, a, b);
}

Value *IDISA_AVX2_Builder::hsimd_packl_in_lanes_impl(unsigned lanes, unsigned fw, Value *a, Value *b) {
    if ((fw == 16) && (lanes == 2)) {
        Function *vpackuswbfunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_packuswb);
        Value *a_low = fwCast(16, simd_and(a, simd_lomask(fw)));
        Value *b_low = fwCast(16, simd_and(b, simd_lomask(fw)));
        return mCB->CreateCall(vpackuswbfunc->getFunctionType(), vpackuswbfunc, {a_low, b_low});
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::hsimd_packl_in_lanes_impl(lanes, fw, a, b);
}

Value *IDISA_AVX2_Builder::hsimd_packh_in_lanes_impl(unsigned lanes, unsigned fw, Value *a, Value *b) {
    if ((fw == 16) && (lanes == 2)) {
        Function *vpackuswbfunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_packuswb);
        Value *a_low = simd_srli(fw, a, fw / 2);
        Value *b_low = simd_srli(fw, b, fw / 2);
        return mCB->CreateCall(vpackuswbfunc->getFunctionType(), vpackuswbfunc, {a_low, b_low});
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::hsimd_packh_in_lanes_impl(lanes, fw, a, b);
}

Value *IDISA_AVX2_Builder::hsimd_packus_impl(unsigned fw, Value *a, Value *b) {
    if (((fw == 32) || (fw == 16)) && (getVectorBitWidth(a) == AVX_width)) {
        Function *pack_func = Intrinsic::getOrInsertDeclaration(
            mCB->getModule(), fw == 16 ? Intrinsic::x86_avx2_packuswb : Intrinsic::x86_avx2_packusdw);
        Value *packed =
            fwCast(64, mCB->CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)}));
        auto field_count = AVX_width / 64;
        SmallVector<Constant *, 4> Idxs(field_count);
        for (unsigned int i = 0; i < field_count / 2; i++) {
            Idxs[i] = mCB->getInt32(2 * i);
            Idxs[i + field_count / 2] = mCB->getInt32(2 * i + 1);
        }
        Constant *shuffleMask = ConstantVector::get(Idxs);
        return mCB->CreateShuffleVector(packed, UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_AVX_Builder::hsimd_packus_impl(fw, a, b);
}

Value *IDISA_AVX2_Builder::hsimd_packss_impl(unsigned fw, Value *a, Value *b) {
    if (((fw == 32) || (fw == 16)) && (getVectorBitWidth(a) == AVX_width)) {
        Function *pack_func = Intrinsic::getOrInsertDeclaration(
            mCB->getModule(), fw == 16 ? Intrinsic::x86_avx2_packsswb : Intrinsic::x86_avx2_packssdw);
        Value *packed =
            fwCast(64, mCB->CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)}));
        auto field_count = AVX_width / 64;
        SmallVector<Constant *, 4> Idxs(field_count);
        for (unsigned int i = 0; i < field_count / 2; i++) {
            Idxs[i] = mCB->getInt32(2 * i);
            Idxs[i + field_count / 2] = mCB->getInt32(2 * i + 1);
        }
        Constant *shuffleMask = ConstantVector::get(Idxs);
        return mCB->CreateShuffleVector(packed, UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_AVX_Builder::hsimd_packss_impl(fw, a, b);
}

std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_add_with_carry_impl(Value *e1, Value *e2, Value *carryin) {
    // using LONG_ADD
    if (getVectorBitWidth(e1) == AVX_width) {
        Type *carryTy = carryin->getType();
        if (carryTy == getBitBlockType()) {
            carryin = mvmd_extract(32, carryin, 0);
        } else if (carryTy->getIntegerBitWidth() < 32) {
            assert(carryTy->isIntegerTy());
            carryin = mCB->CreateZExt(carryin, mCB->getInt32Ty());
        }
        Value *carrygen = simd_and(e1, e2);
        Value *carryprop = simd_or(e1, e2);
        Value *digitsum = simd_add(64, e1, e2);
        Value *digitcarry = simd_or(carrygen, simd_and(carryprop, mCB->CreateNot(digitsum)));
        Value *carryMask = hsimd_signmask(64, digitcarry);
        Value *carryMask2 = mCB->CreateOr(mCB->CreateAdd(carryMask, carryMask), carryin);
        Value *bubble = simd_eq(64, digitsum, allOnes());
        Value *bubbleMask = hsimd_signmask(64, bubble);
        Value *incrementMask = mCB->CreateXor(mCB->CreateAdd(bubbleMask, carryMask2), bubbleMask);
        Value *increments = esimd_bitspread(AVX_width, 64, mCB->CreateTrunc(incrementMask, mCB->getIntNTy(4)));
        Value *sum = simd_add(64, digitsum, increments);
        Value *carry_out = mCB->CreateLShr(incrementMask, AVX_width / 64);
        assert(carry_out->getType()->getIntegerBitWidth() == 32);
        if (carryTy == e1->getType()) {
            carry_out = mCB->CreateZExtOrTrunc(carry_out, getBitBlockType()->getElementType());
            carry_out =
                mCB->CreateInsertElement(ConstantVector::getNullValue(e1->getType()), carry_out, mCB->getInt32(0));
        } else if (carryTy != carry_out->getType()) {
            carry_out = mCB->CreateZExtOrTrunc(carry_out, carryTy);
        }
        return std::pair<Value *, Value *>{carry_out, bitCast(sum)};
    }
    return IDISA_AVX_Builder::bitblock_add_with_carry_impl(e1, e2, carryin);
}

std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_advance_impl(Value *a, Value *shiftin, unsigned shift) {
    if (getVectorBitWidth(a) == AVX_width) {
        if (shiftin->getType() == mCB->getInt8Ty() && shift == 1) {
            const uint32_t fw = AVX_width / 8;
            Type *const v32xi8Ty = FixedVectorType::get(mCB->getInt8Ty(), 32);
            Type *const v32xi32Ty = FixedVectorType::get(mCB->getInt32Ty(), 32);
            Value *shiftin_block = mCB->CreateInsertElement(Constant::getNullValue(v32xi8Ty), shiftin, (uint64_t)0);
            shiftin_block =
                mCB->CreateShuffleVector(shiftin_block, UndefValue::get(v32xi8Ty), Constant::getNullValue(v32xi32Ty));
            shiftin_block = bitCast(shiftin_block);
            Value *field_shift = bitCast(mvmd_dslli(fw, a, shiftin_block, 1));
            Value *shifted = bitCast(mCB->CreateOr(mCB->CreateLShr(field_shift, fw - shift), mCB->CreateShl(a, shift)));
            Value *shiftout = hsimd_signmask(fw, a);
            shiftout = mCB->CreateTrunc(shiftout, mCB->getInt8Ty());
            shiftout = mCB->CreateAnd(shiftout, mCB->getInt8(0x80));
            return std::make_pair(shiftout, shifted);
        }
    }
    return IDISA_AVX_Builder::bitblock_advance_impl(a, shiftin, shift);
}

std::vector<Value *> IDISA_AVX2_Builder::simd_pext_impl(unsigned fw, std::vector<Value *> vs, Value *extract_mask) {
    if (mCB->hasFeature(codegen::Feature::AVX_BMI2) && (fw <= 64)) {
        unsigned fieldCount = getVectorBitWidth(vs[0]) / fw;
        SmallVector<Value *> mask(fieldCount);
        for (unsigned i = 0; i < fieldCount; i++) {
            mask[i] = mvmd_extract(fw, extract_mask, i);
            if (fw < 32) {
                mask[i] = mCB->CreateZExt(mask[i], mCB->getInt32Ty());
            }
        }
        Type *fwTy = mCB->getIntNTy(fw);
        std::vector<Value *> results;
        for (Value *v : vs) {
            assert((getVectorBitWidth(v) == fw * fieldCount) && "Vectors must have uniform size");
            Value *res;
            if (fw == 64) {
                res = vectorize(fw, v, [=](unsigned i, Value *v_i) -> Value * {
                    return mCB->CreateIntrinsic(Intrinsic::x86_bmi_pext_64, {v_i, mask[i]});
                });
            } else if (fw == 32) {
                res = vectorize(fw, v, [=](unsigned i, Value *v_i) -> Value * {
                    return mCB->CreateIntrinsic(Intrinsic::x86_bmi_pext_32, {v_i, mask[i]});
                });
            } else {
                res = vectorize(fw, v, [=](unsigned i, Value *v_i) -> Value * {
                    v_i = mCB->CreateZExt(v_i, mCB->getInt32Ty());
                    Value *r = mCB->CreateIntrinsic(Intrinsic::x86_bmi_pext_32, {v_i, mask[i]});
                    return mCB->CreateTrunc(r, fwTy);
                });
            }
            results.push_back(res);
        }
        return results;
    }
    return IDISA_AVX_Builder::simd_pext_impl(fw, vs, extract_mask);
}

Value *IDISA_AVX2_Builder::simd_pdep_impl(unsigned fw, Value *v, Value *deposit_mask) {
    if (mCB->hasFeature(codegen::Feature::AVX_BMI2) && (fw <= 64)) {
        if (fw == 64) {
            return vectorize(fw, v, deposit_mask, [=](unsigned i, Value *v_i, Value *mask_i) -> Value * {
                return mCB->CreateIntrinsic(Intrinsic::x86_bmi_pdep_64, {v_i, mask_i});
            });
        } else if (fw == 32) {
            return vectorize(fw, v, deposit_mask, [=](unsigned i, Value *v_i, Value *mask_i) -> Value * {
                return mCB->CreateIntrinsic(Intrinsic::x86_bmi_pdep_32, {v_i, mask_i});
            });
        } else {
            Type *fwTy = mCB->getIntNTy(fw);
            return vectorize(fw, v, deposit_mask, [=](unsigned i, Value *v_i, Value *mask_i) -> Value * {
                v_i = mCB->CreateZExt(v_i, mCB->getInt32Ty());
                mask_i = mCB->CreateZExt(mask_i, mCB->getInt32Ty());
                Value *r = mCB->CreateIntrinsic(Intrinsic::x86_bmi_pdep_32, {v_i, mask_i});
                return mCB->CreateTrunc(r, fwTy);
            });
        }
    }
    return IDISA_AVX_Builder::simd_pdep_impl(fw, v, deposit_mask);
}

std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_indexed_advance_impl(Value *strm, Value *index_strm,
                                                                              Value *shiftIn, unsigned shiftAmount) {
    const unsigned bitWidth = mCB->getSizeTy()->getBitWidth();
    if (mCB->hasFeature(codegen::Feature::AVX_BMI2) && ((bitWidth == 64) || (bitWidth == 32))) {
        Function *PEXT_f = (bitWidth == 64)
                               ? Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_64)
                               : Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_32);
        Function *PDEP_f = (bitWidth == 64)
                               ? Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_64)
                               : Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_32);
        Function *const popcount =
            Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::ctpop, mCB->getSizeTy());
        Type *iBitBlock = mCB->getIntNTy(getBitBlockWidth());
        Value *shiftVal = mCB->getSize(shiftAmount);
        const auto n = getBitBlockWidth() / bitWidth;
        FixedVectorType *const vecTy = FixedVectorType::get(mCB->getSizeTy(), n);
        if (LLVM_LIKELY(shiftAmount < bitWidth)) {
            Value *carry = mvmd_extract(bitWidth, shiftIn, 0);
            Value *result = UndefValue::get(vecTy);
            for (unsigned i = 0; i < n; i++) {
                Value *s = mvmd_extract(bitWidth, strm, i);
                Value *ix = mvmd_extract(bitWidth, index_strm, i);
                Value *ix_popcnt = mCB->CreateCall(popcount->getFunctionType(), popcount, {ix});
                Value *bits = mCB->CreateCall(PEXT_f->getFunctionType(), PEXT_f, {s, ix});
                Value *adv = mCB->CreateOr(mCB->CreateShl(bits, shiftAmount), carry);
                // We have two cases depending on whether the popcount of the index pack is < shiftAmount or not.
                Value *popcount_small = mCB->CreateICmpULT(ix_popcnt, shiftVal);
                Value *carry_if_popcount_small = mCB->CreateOr(
                    mCB->CreateShl(bits, mCB->CreateSub(shiftVal, ix_popcnt)), mCB->CreateLShr(carry, ix_popcnt));
                Value *carry_if_popcount_large = mCB->CreateLShr(bits, mCB->CreateSub(ix_popcnt, shiftVal));
                carry = mCB->CreateSelect(popcount_small, carry_if_popcount_small, carry_if_popcount_large);
                result =
                    mvmd_insert(bitWidth, result, mCB->CreateCall(PDEP_f->getFunctionType(), PDEP_f, {adv, ix}), i);
            }
            Value *carryOut = mvmd_insert(bitWidth, allZeroes(), carry, 0);
            return std::pair<Value *, Value *>{bitCast(carryOut), bitCast(result)};
        } else if (shiftAmount <= getBitBlockWidth()) {
            // The shift amount is always greater than the popcount of the individual
            // elements that we deal with.   This simplifies some of the logic.
            Value *carry = mCB->CreateBitCast(shiftIn, iBitBlock);
            Value *result = UndefValue::get(vecTy);
            for (unsigned i = 0; i < n; i++) {
                Value *s = mvmd_extract(bitWidth, strm, i);
                Value *ix = mvmd_extract(bitWidth, index_strm, i);
                Value *ix_popcnt = mCB->CreateCall(popcount->getFunctionType(), popcount, {ix});
                Value *bits = mCB->CreateCall(PEXT_f->getFunctionType(), PEXT_f,
                                              {s, ix}); // All these bits are shifted out (appended to carry).
                result = mvmd_insert(
                    bitWidth, result,
                    mCB->CreateCall(PDEP_f->getFunctionType(), PDEP_f, {mvmd_extract(bitWidth, carry, 0), ix}), i);
                carry = mCB->CreateLShr(
                    carry,
                    mCB->CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed, make room for new bits.
                carry = mCB->CreateOr(carry,
                                      mCB->CreateShl(mCB->CreateZExt(bits, iBitBlock),
                                                     mCB->CreateZExt(mCB->CreateSub(shiftVal, ix_popcnt), iBitBlock)));
            }
            return std::pair<Value *, Value *>{bitCast(carry), bitCast(result)};
        } else {
            // The shift amount is greater than the total popcount.   We will consume popcount
            // bits from the shiftIn value only, and produce a carry out value of the selected bits.
            // elements that we deal with.   This simplifies some of the logic.
            Value *carry = mCB->CreateBitCast(shiftIn, iBitBlock);
            Value *result = UndefValue::get(vecTy);
            Value *carryOut = ConstantInt::getNullValue(iBitBlock);
            Value *generated = mCB->getSize(0);
            for (unsigned i = 0; i < n; i++) {
                Value *s = mvmd_extract(bitWidth, strm, i);
                Value *ix = mvmd_extract(bitWidth, index_strm, i);
                Value *ix_popcnt = mCB->CreateCall(popcount->getFunctionType(), popcount, {ix});
                Value *bits = mCB->CreateCall(PEXT_f->getFunctionType(), PEXT_f,
                                              {s, ix}); // All these bits are shifted out (appended to carry).
                result = mvmd_insert(
                    bitWidth, result,
                    mCB->CreateCall(PDEP_f->getFunctionType(), PDEP_f, {mvmd_extract(bitWidth, carry, 0), ix}), i);
                carry =
                    mCB->CreateLShr(carry, mCB->CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed.
                carryOut = mCB->CreateOr(
                    carryOut, mCB->CreateShl(mCB->CreateZExt(bits, iBitBlock), mCB->CreateZExt(generated, iBitBlock)));
                generated = mCB->CreateAdd(generated, ix_popcnt);
            }
            return std::pair<Value *, Value *>{bitCast(carryOut), bitCast(result)};
        }
    }
    return IDISA_AVX_Builder::bitblock_indexed_advance_impl(strm, index_strm, shiftIn, shiftAmount);
}

Value *IDISA_AVX2_Builder::hsimd_signmask_impl(unsigned fw, Value *a) {
    // AVX2 special cases
    if (getVectorBitWidth(a) == AVX_width) {
        if (fw == 8) {
            Function *signmask_f8func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_pmovmskb);
            return mCB->CreateCall(signmask_f8func->getFunctionType(), signmask_f8func, fwCast(8, a));
        }
    }
    // Otherwise use default SSE logic.
    return IDISA_AVX_Builder::hsimd_signmask_impl(fw, a);
}

Value *IDISA_AVX2_Builder::mvmd_srl_impl(unsigned fw, Value *a, Value *shift, const bool safe) {
    // Intrinsic::x86_avx2_permd) allows an efficient implementation for field width 32.
    // Translate larger field widths to 32 bits.
    if (LLVM_LIKELY(getVectorBitWidth(a) == AVX_width)) {
        if (fw > 32) {
            return fwCast(fw,
                          mvmd_srl(32, a, mCB->CreateMul(shift, ConstantInt::get(shift->getType(), fw / 32)), safe));
        } else if (fw == 32) {
            Function *permuteFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_permd);
            const unsigned fieldCount = AVX_width / fw;
            Type *fieldTy = mCB->getIntNTy(fw);
            SmallVector<Constant *, 16> indexes(fieldCount);
            for (unsigned int i = 0; i < fieldCount; i++) {
                indexes[i] = ConstantInt::get(fieldTy, i);
            }
            Constant *indexVec = ConstantVector::get(indexes);
            Constant *fieldCountSplat = getSplat(fieldCount, ConstantInt::get(fieldTy, fieldCount));
            Value *shiftSplat = simd_fill(fw, mCB->CreateZExtOrTrunc(shift, fieldTy));
            Value *permuteVec = mCB->CreateAdd(indexVec, shiftSplat);
            // Zero out fields that are above the max.
            permuteVec = simd_and(permuteVec, simd_ult(fw, permuteVec, fieldCountSplat));
            // Insert a zero value at position 0 (OK for shifts > 0)
            Value *a0 = mvmd_insert(fw, a, Constant::getNullValue(fieldTy), 0);
            Value *shifted = mCB->CreateCall(permuteFunc->getFunctionType(), permuteFunc, {a0, permuteVec});
            return fwCast(32, simd_if(1, simd_eq(fw, shiftSplat, allZeroes()), a, shifted));
        }
    }
    return IDISA_AVX_Builder::mvmd_srl_impl(fw, a, shift, safe);
}

Value *IDISA_AVX2_Builder::mvmd_sll_impl(unsigned fw, Value *a, Value *shift, const bool safe) {
    // Intrinsic::x86_avx2_permd) allows an efficient implementation for field width 32.
    // Translate larger field widths to 32 bits.
    if (getVectorBitWidth(a) == AVX_width) {
        if (fw > 32) {
            return fwCast(fw,
                          mvmd_sll(32, a, mCB->CreateMul(shift, ConstantInt::get(shift->getType(), fw / 32)), safe));
        } else if (fw == 32) {
            Function *permuteFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_permd);
            const unsigned fieldCount = AVX_width / fw;
            Type *fieldTy = mCB->getIntNTy(fw);
            SmallVector<Constant *, 16> indexes(fieldCount);
            for (unsigned int i = 0; i < fieldCount; i++) {
                indexes[i] = ConstantInt::get(fieldTy, i);
            }
            Constant *indexVec = ConstantVector::get(indexes);
            Value *shiftSplat = simd_fill(fw, mCB->CreateZExtOrTrunc(shift, fieldTy));
            Value *permuteVec = mCB->CreateSub(indexVec, shiftSplat);
            // Negative indexes are for fields that must be zeroed.  Convert the
            // permute constant to an all ones value, that will select item 7.
            permuteVec = simd_or(permuteVec, simd_lt(fw, permuteVec, fwCast(fw, allZeroes())));
            // Insert a zero value at position 7 (OK for shifts > 0)
            Value *a0 = mvmd_insert(fw, a, Constant::getNullValue(fieldTy), 7);
            Value *shifted = mCB->CreateCall(permuteFunc->getFunctionType(), permuteFunc, {a0, permuteVec});
            return fwCast(32, simd_if(1, simd_eq(fw, shiftSplat, allZeroes()), a, shifted));
        }
    }
    return IDISA_AVX_Builder::mvmd_sll_impl(fw, a, shift, safe);
}

Value *IDISA_AVX2_Builder::mvmd_shuffle_impl(unsigned fw, Value *a, Value *index_vector, ShuffleMode mode) {
    if (getVectorBitWidth(a) == AVX_width && (fw == 32)) {
        auto fieldCount = AVX_width / fw;
        // x86_avx2_permd truncates indices to 3 bits, does not zero.
        Function *shuf32Func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx2_permd);
        Value *shuf =
            mCB->CreateCall(shuf32Func->getFunctionType(), shuf32Func, {fwCast(32, a), fwCast(32, index_vector)});
        if (mode == ShuffleMode::ZeroOnHighIndexBit) {
            Constant *high_bit = Constant::getIntegerValue(mCB->getInt32Ty(), APInt::getHighBitsSet(fw, 1));
            Value *enable_shuf = simd_ult(fw, index_vector, getSplat(fieldCount, high_bit));
            return simd_and(shuf, enable_shuf);
        } else if (mode == ShuffleMode::ZeroOnIndexOver) {
            Value *enable_shuf = simd_ult(fw, index_vector, getSplat(fieldCount, mCB->getInt32(fieldCount)));
            return simd_and(shuf, enable_shuf);
        }
        return shuf; // if (mode == ShuffleMode::TruncateIndex)
    }
    if (getVectorBitWidth(a) == AVX_width && (fw == 8)) {
        // x86_avx2_pshuf_b shuffles within 128 bit lanes, zeroing if the high bit is set.
        constexpr unsigned fieldCount = AVX_width / 8;

        if (mode == ShuffleMode::TruncateIndex) {
            // Clear high bits
            index_vector = simd_and(index_vector, getSplatN(8, fieldCount, fieldCount - 1));
        } else if (mode == ShuffleMode::ZeroOnIndexOver) {
            Value *over = simd_ugt(fw, index_vector, getSplatN(8, fieldCount, fieldCount - 1));
            index_vector = simd_or(index_vector, over);
        }

        VectorType *vec64Ty = fwVectorType(64);
        Value *const a64 = mCB->CreateBitCast(a, vec64Ty);
        VectorType *vecTy = fwVectorType(8);
        index_vector = mCB->CreateBitCast(index_vector, vecTy);

        // Because the shuffle only happens within 128-bit lanes, we do it twice, once to select the elements we want
        // from the bottom half of the 256-bit word, and a second time to select the elements from the top half. Then,
        // OR those results together.
        Constant *hiBits = getSplat(fieldCount, APInt::getHighBitsSet(8, 1));
        Value *isHiIdxs = simd_and(simd_slli(fw, index_vector, 3), hiBits);

        Value *loSelect = ConstantVector::get({mCB->getInt64(0), mCB->getInt64(1), mCB->getInt64(0), mCB->getInt64(1)});
        Value *loVec = fwCast(fw, mCB->CreateShuffleVector(a64, UndefValue::get(vec64Ty), loSelect));
        Value *loShufIdxs = simd_or(index_vector, isHiIdxs);

        Value *hiSelect = ConstantVector::get({mCB->getInt64(2), mCB->getInt64(3), mCB->getInt64(2), mCB->getInt64(3)});
        Value *hiVec = fwCast(fw, mCB->CreateShuffleVector(a64, UndefValue::get(vec64Ty), hiSelect));
        Value *hiShufIdxs = simd_or(index_vector, simd_xor(isHiIdxs, hiBits));
        
        Value *loShuffle = mCB->CreateIntrinsic(Intrinsic::x86_avx2_pshuf_b, {loVec, fwCast(fw, loShufIdxs)});
        Value *hiShuffle = mCB->CreateIntrinsic(Intrinsic::x86_avx2_pshuf_b, {hiVec, fwCast(fw, hiShufIdxs)});
        return fwCast(fw, simd_or(loShuffle, hiShuffle));
    }
    return IDISA_AVX_Builder::mvmd_shuffle_impl(fw, a, index_vector, mode);
}

llvm::Value *IDISA_AVX2_Builder::mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                                    llvm::Value *index_vector, ShuffleMode mode) {
    return IDISA_AVX_Builder::mvmd_shuffle2_impl(fw, table0, table1, index_vector, mode);
}

Value *IDISA_AVX2_Builder::mvmd_compress_impl(unsigned fw, Value *a, Value *select_mask) {
    if (mCB->hasFeature(codegen::Feature::AVX_BMI2) && (getVectorBitWidth(a) == AVX_width)) {
        if (fw == 64) {
            Function *PDEP_func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_32);
            Value *mask = mCB->CreateZExtOrTrunc(select_mask, mCB->getInt32Ty());
            Value *mask32 =
                mCB->CreateMul(mCB->CreateCall(PDEP_func->getFunctionType(), PDEP_func, {mask, mCB->getInt32(0x55)}),
                               mCB->getInt32(3));
            Value *result = fwCast(fw, mvmd_compress(32, fwCast(32, a), mCB->CreateTrunc(mask32, mCB->getInt8Ty())));
            return result;
        } else if (fw == 32) {
            Type *v1xi32Ty = FixedVectorType::get(mCB->getInt32Ty(), 1);
            Type *v8xi32Ty = FixedVectorType::get(mCB->getInt32Ty(), 8);
            Type *v8xi1Ty = FixedVectorType::get(mCB->getInt1Ty(), 8);
            Constant *mask0000000Fsplaat = getSplat(8, mCB->getInt32(0xF));
            Function *PEXT_func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_32);
            Function *PDEP_func = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_32);
            Function *const popcount_func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::ctpop, mCB->getInt32Ty());
            // First duplicate each mask bit to select 4-bit fields
            Value *mask = mCB->CreateZExtOrTrunc(select_mask, mCB->getInt32Ty());
            Value *field_count = mCB->CreateCall(popcount_func->getFunctionType(), popcount_func, mask);
            assert(field_count->getType()->getIntegerBitWidth() >= 32);
            Value *spread = mCB->CreateCall(PDEP_func->getFunctionType(), PDEP_func, {mask, mCB->getInt32(0x11111111)});
            Value *ext_mask = mCB->CreateMul(spread, mCB->getInt32(0xF));
            // Now extract the 4-bit index values for the required fields.
            Value *indexes =
                mCB->CreateCall(PEXT_func->getFunctionType(), PEXT_func, {mCB->getInt32(0x76543210), ext_mask});
            // Broadcast to all fields
            Value *bdcst = mCB->CreateShuffleVector(mCB->CreateBitCast(indexes, v1xi32Ty), UndefValue::get(v1xi32Ty),
                                                    ConstantVector::getNullValue(v8xi32Ty));
            Constant *Shifts[8];
            for (unsigned i = 0; i < 8; i++) {
                Shifts[i] = mCB->getInt32(i * 4);
            }
            Value *shuf = mCB->CreateAnd(mCB->CreateLShr(bdcst, ConstantVector::get({Shifts, 8})), mask0000000Fsplaat);

            Value *compress = mvmd_shuffle(32, a, shuf);
            Value *field_mask = mCB->CreateTrunc(
                mCB->CreateSub(mCB->CreateShl(mCB->getInt32(1), field_count), mCB->getInt32(1)), mCB->getInt8Ty());
            return mCB->CreateAnd(compress, mCB->CreateSExt(mCB->CreateBitCast(field_mask, v8xi1Ty), v8xi32Ty));
        } else if (fw >= 8) {

            unsigned fieldCount = 256 / fw;

            // Step 1: Initialize indices as 6-bit bixnum in an array of 64-bit integers
            uint64_t indices[6] = {0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC, 0xF0F0F0F0F0F0F0F0,
                                   0xFF00FF00FF00FF00, 0xFFFF0000FFFF0000, 0xFFFFFFFF00000000};

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected
            Function *pextFunc = nullptr;
            if (LLVM_LIKELY(fieldCount == 64)) {
                pextFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_64);
            } else {
                assert(fieldCount <= 32);
                pextFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_32);
                if (LLVM_UNLIKELY(fieldCount < 32)) {
                    assert(select_mask->getType()->getIntegerBitWidth() <= 32);
                    select_mask = mCB->CreateZExt(select_mask, mCB->getInt32Ty());
                }
            }

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

            Value *permute_vec = nullptr;
            const auto m = floor_log2(fieldCount);
            FixedArray<Value *, 2> args;
            args[1] = select_mask;

            IntegerType *intTy = mCB->getIntNTy((fieldCount == 64) ? 64 : 32);

            Type *resultTy = fwVectorType(fw);

            for (unsigned i = 0; i < m; ++i) {
                args[0] = ConstantInt::get(intTy, indices[i]);
                Value *const expanded = mCB->CreateCall(pextFunc->getFunctionType(), pextFunc, args);
                Value *byteExpanded = esimd_bitspread(AVX_width, fw, expanded);
                if (i == 0) {
                    permute_vec = byteExpanded;
                } else {
                    Value *shiftedExpand = simd_slli(64, byteExpanded, i);
                    permute_vec = simd_or(permute_vec, shiftedExpand);
                }
            }
            // // Step 4: Use mvmd_shuffle2 to shuffle using permute_vec
            Value *const shuffled = mvmd_shuffle(fw, a, permute_vec);
            Value *const count = mCB->CreatePopcount(select_mask);
            Constant *ALL_ONES = ConstantVector::getAllOnesValue(resultTy);
            Value *mask = mCB->CreateNot(mvmd_sll(fw, ALL_ONES, count));
            mask = mCB->CreateSelect(mCB->CreateICmpNE(count, ConstantInt::get(intTy, fieldCount)), mask, ALL_ONES);
            assert(shuffled->getType() == mask->getType());
            return mCB->CreateAnd(shuffled, mask);
        }
    }

    return IDISA_AVX_Builder::mvmd_compress_impl(fw, a, select_mask);
}

Value *IDISA_AVX2_Builder::mvmd_expand_impl(unsigned fw, Value *a, Value *select_mask) {
// Generic mvmd_expand is faster
#if 0
    if (mCB->hasFeature(codegen::Feature::AVX_BMI2) && (getVectorBitWidth(a) == AVX_width)) {
         if (fw >= 8) {

             const auto fieldCount = 256 / fw;

             uint64_t indices[6] = {
                 0xAAAAAAAAAAAAAAAA,
                 0xCCCCCCCCCCCCCCCC,
                 0xF0F0F0F0F0F0F0F0,
                 0xFF00FF00FF00FF00,
                 0xFFFF0000FFFF0000,
                 0xFFFFFFFF00000000
             };

             Function * pdepFunc = nullptr;
             if (LLVM_LIKELY(fieldCount == 64)) {
                 pdepFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_64);
             } else {
                 assert (fieldCount <= 32);
                 pdepFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_32);
                 if (LLVM_UNLIKELY(fieldCount < 32)) {
                     assert (select_mask->getType()->getIntegerBitWidth() <= 32);
                     select_mask = mCB->CreateZExt(select_mask, mCB->getInt32Ty());
                 }
             }

             // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

             Value * permute_vec = nullptr;
             const auto m = floor_log2(fieldCount);
             FixedArray<Value *, 2> args;
             args[1] = select_mask;

             IntegerType * intTy = mCB->getIntNTy((fieldCount == 64) ? 64 : 32);
             for (unsigned i = 0; i < m; ++i) {
                 args[0] = ConstantInt::get(intTy, indices[i]);
                 Value * const expanded = mCB->CreateCall(pdepFunc->getFunctionType(), pdepFunc, args);
                 Value * byteExpanded = esimd_bitspread(AVX_width, fw, expanded);
                 if (i == 0) {
                     permute_vec = byteExpanded;
                 } else {
                     Value * shiftedExpand = simd_slli(64, byteExpanded, i);
                     permute_vec = simd_or(permute_vec, shiftedExpand);
                 }
             }

             // // Step 4: Use mvmd_shuffle to shuffle using permute_vec
             Value * const shuffled = mvmd_shuffle(fw, a, permute_vec);
             Value * const mask = simd_any(fw, esimd_bitspread(AVX_width, fw, select_mask));
             assert (shuffled->getType() == mask->getType());
             return mCB->CreateAnd(shuffled, mask);
         }
    }
#endif
    return IDISA_AVX_Builder::mvmd_expand_impl(fw, a, select_mask);
}

DEFINE_BUILDER_CACHE_NAME(IDISA_AVX512F_Builder, "AVX512F", AVX512_width)

Value *IDISA_AVX512F_Builder::hsimd_packh_impl(unsigned fw, Value *a, Value *b) {
    if ((getVectorBitWidth(a) == AVX512_width) && (fw == 16)) {
        a = fwCast(fw, a);
        a = simd_srli(fw, a, fw / 2);
        b = fwCast(fw, b);
        b = simd_srli(fw, b, fw / 2);
        return hsimd_packl(fw, a, b);
    }
    return IDISA_AVX2_Builder::hsimd_packh_impl(fw, a, b);
}

Value *IDISA_AVX512F_Builder::hsimd_packl_impl(unsigned fw, Value *a, Value *b) {
    if ((getVectorBitWidth(a) == AVX512_width) && (fw == 16)) {

        const unsigned int field_count = 64;
        Constant *Idxs[field_count];
        for (unsigned int i = 0; i < field_count; i++) {
            Idxs[i] = mCB->getInt32(i);
        }
        Constant *shuffleMask = ConstantVector::get({Idxs, 64});
        Value *a1 = mCB->CreateTrunc(fwCast(fw, a), FixedVectorType::get(mCB->getInt8Ty(), 32));
        Value *b1 = mCB->CreateTrunc(fwCast(fw, b), FixedVectorType::get(mCB->getInt8Ty(), 32));
        return mCB->CreateShuffleVector(a1, b1, shuffleMask);
    }
    return IDISA_AVX2_Builder::hsimd_packl_impl(fw, a, b);
}

Value *IDISA_AVX512F_Builder::hsimd_packus_impl(unsigned fw, Value *a, Value *b) {
    if (mCB->hasFeature(codegen::Feature::AVX512_BW) && ((fw == 16) || (fw == 32)) &&
        (getVectorBitWidth(a) == AVX512_width)) {
        Function *pack_func = Intrinsic::getOrInsertDeclaration(
            mCB->getModule(), fw == 16 ? Intrinsic::x86_avx512_packuswb_512 : Intrinsic::x86_avx512_packusdw_512);
        Value *packed = mCB->CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)});
        auto field_count = AVX512_width / 64;
        SmallVector<Constant *, 16> Idxs(field_count);
        for (unsigned int i = 0; i < field_count / 2; i++) {
            Idxs[i] = mCB->getInt32(2 * i);
            Idxs[i + field_count / 2] = mCB->getInt32(2 * i + 1);
        }
        Constant *shuffleMask = ConstantVector::get(Idxs);
        return mCB->CreateShuffleVector(fwCast(64, packed), UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_AVX2_Builder::hsimd_packus_impl(fw, a, b);
}

Value *IDISA_AVX512F_Builder::hsimd_packss_impl(unsigned fw, Value *a, Value *b) {
    if (mCB->hasFeature(codegen::Feature::AVX512_BW) && ((fw == 16) || (fw == 32)) &&
        (getVectorBitWidth(a) == AVX512_width)) {
        Function *pack_func = Intrinsic::getOrInsertDeclaration(
            mCB->getModule(), fw == 16 ? Intrinsic::x86_avx512_packsswb_512 : Intrinsic::x86_avx512_packssdw_512);
        Value *packed = mCB->CreateCall(pack_func->getFunctionType(), pack_func, {fwCast(fw, a), fwCast(fw, b)});
        auto field_count = AVX512_width / 64;
        SmallVector<Constant *, 16> Idxs(field_count);
        for (unsigned int i = 0; i < field_count / 2; i++) {
            Idxs[i] = mCB->getInt32(2 * i);
            Idxs[i + field_count / 2] = mCB->getInt32(2 * i + 1);
        }
        Constant *shuffleMask = ConstantVector::get(Idxs);
        return mCB->CreateShuffleVector(fwCast(64, packed), UndefValue::get(fwVectorType(64)), shuffleMask);
    }
    // Otherwise use default logic.
    return IDISA_AVX2_Builder::hsimd_packus_impl(fw, a, b);
}

Value *IDISA_AVX512F_Builder::mvmd_srl_impl(unsigned fw, Value *a, Value *shift, const bool safe) {
    if (getVectorBitWidth(a) == AVX512_width && fw >= 8) {
        const auto fieldCount = 512 / fw;
        Type *fieldTy = mCB->getIntNTy(fw);
        SmallVector<Constant *, 64> indexes(fieldCount);
        for (unsigned i = 0; i < fieldCount; i++) {
            indexes[i] = ConstantInt::get(fieldTy, i);
        }
        Constant *indexVec = ConstantVector::get(indexes);
        Value *broadcast = simd_fill(fw, mCB->CreateZExtOrTrunc(shift, fieldTy));
        Value *permuteVec = mCB->CreateAdd(indexVec, broadcast);
        return mvmd_shuffle2(fw, fwCast(fw, a), fwCast(fw, allZeroes()), permuteVec);
    }
    return IDISA_AVX2_Builder::mvmd_srl_impl(fw, a, shift, safe);
}

Value *IDISA_AVX512F_Builder::mvmd_sll_impl(unsigned fw, Value *a, Value *shift, const bool safe) {
    if (getVectorBitWidth(a) == AVX512_width && fw >= 8) {
        const auto fieldCount = 512 / fw;
        Type *fieldTy = mCB->getIntNTy(fw);
        SmallVector<Constant *, 64> indexes(fieldCount);
        for (unsigned i = 0; i < fieldCount; i++) {
            indexes[i] = ConstantInt::get(fieldTy, fieldCount + i);
        }
        Constant *indexVec = ConstantVector::get(indexes);
        Value *broadcast = simd_fill(fw, mCB->CreateZExtOrTrunc(shift, fieldTy));
        Value *permuteVec = mCB->CreateSub(indexVec, broadcast);
        return mvmd_shuffle2(fw, fwCast(fw, allZeroes()), fwCast(fw, a), permuteVec);
    }
    return IDISA_AVX2_Builder::mvmd_sll_impl(fw, a, shift);
}

Value *IDISA_AVX512F_Builder::mvmd_shuffle_impl(unsigned fw, Value *data_table, Value *index_vector, ShuffleMode mode) {
    if (getVectorBitWidth(data_table) == AVX512_width) {
        auto fieldCount = AVX512_width / fw;
        Type *fwTy = mCB->getIntNTy(fw);
        Value *shuf = mvmd_shuffle2(fw, data_table, data_table, index_vector, ShuffleMode::TruncateIndex);
        if (mode == ShuffleMode::ZeroOnHighIndexBit) {
            Constant *high_bit = Constant::getIntegerValue(fwTy, APInt::getHighBitsSet(fw, 1));
            Value *enable_shuf = simd_ult(fw, index_vector, getSplat(fieldCount, high_bit));
            return simd_and(shuf, enable_shuf);
        } else if (mode == ShuffleMode::ZeroOnIndexOver) {
            Value *enable_shuf = simd_ult(fw, index_vector, getSplat(fieldCount, ConstantInt::get(fwTy, fieldCount)));
            return simd_and(shuf, enable_shuf);
        }
        return shuf;
    }
    return IDISA_AVX2_Builder::mvmd_shuffle_impl(fw, data_table, index_vector, mode);
}

#define AVX512_MASK_PERMUTE_INTRINSIC(i) Intrinsic::x86_avx512_vpermi2##i

Value *IDISA_AVX512F_Builder::mvmd_shuffle2_impl(unsigned fw, Value *table0, Value *table1, Value *index_vector,
                                                 ShuffleMode mode) {
    if (getVectorBitWidth(table0) == AVX512_width) {
        auto fieldCount = 2 * AVX512_width / fw;
        Type *fwTy = mCB->getIntNTy(fw);
        Function *permuteFunc = nullptr;
        // First consider the family of x86_avx512_vpermi2 intrinsics
        if (fw == 32) {
            permuteFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_d_512));
        } else if (fw == 64) {
            permuteFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_q_512));
        } else if (fw == 16 && mCB->hasFeature(codegen::Feature::AVX512_BW)) {
            permuteFunc =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_hi_512));
        } else if (fw == 8 && mCB->hasFeature(codegen::Feature::AVX512_VBMI)) {
            permuteFunc =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), AVX512_MASK_PERMUTE_INTRINSIC(var_qi_512));
        }
        if (permuteFunc) {
            Value *shuf = mCB->CreateCall(permuteFunc->getFunctionType(), permuteFunc,
                                          {fwCast(fw, table0), fwCast(fw, index_vector), fwCast(fw, table1)});
            if (mode == ShuffleMode::ZeroOnHighIndexBit) {
                Value *enable_shuf =
                    simd_ult(fw, index_vector, getSplat(fieldCount, ConstantInt::get(fwTy, 1 << (fw - 1))));
                return simd_and(shuf, enable_shuf);
            } else if (mode == ShuffleMode::ZeroOnIndexOver) {
                Value *enable_shuf =
                    simd_ult(fw, index_vector, getSplat(fieldCount, ConstantInt::get(fwTy, fieldCount)));
                return simd_and(shuf, enable_shuf);
            }
            return shuf; // if (mode == ShuffleMode::TruncateIndex)
        }
        if (fw == 8 && mCB->hasFeature(codegen::Feature::AVX512_BW)) {

            // If we have AVX512BW but not AVX512VBMI, we can use 16 bit shuffles to replicate an 8 bit shuffle.
            // This requires us to split the table look up into a lower and higher "half" table and index vectors
            // since we need to zero extend each field.

            // Although the tables can be easily zero extended, the indices are a bit trickier since we could have:
            // <0, 63, 1, 62, ...> as a pattern. Thus we build up both the results from selecting the lower and higher
            // tables separately then OR them together.

            VectorType *vty = fwVectorType(8);

            assert(index_vector->getType() == vty);

            Constant *const ZEROES = ConstantVector::getNullValue(vty);

#define ZEXT16L(T) esimd_mergel(8, (T), ZEROES)
#define ZEXT16H(T) esimd_mergeh(8, (T), ZEROES)

            Constant *const ALL_64 = getSplat(512 / 8, mCB->getInt8(64));
            Value *const InL = simd_lt(8, index_vector, ALL_64);
            assert(InL->getType() == vty);
            Value *IndexVectorL = mCB->CreateAnd(index_vector, InL);
            Value *const InH = mCB->CreateNot(InL);
            Value *IndexVectorH = mCB->CreateAnd(mCB->CreateSub(index_vector, ALL_64), InH);
            Value *IndexVector = mCB->CreateOr(IndexVectorL, IndexVectorH);
            Value *const IL = ZEXT16L(IndexVector);
            Value *const IH = ZEXT16H(IndexVector);
            auto SelectFromHalfTable = [&](Value *table, Value *Mask) {
                Value *T0 = ZEXT16L(table);
                Value *T1 = ZEXT16H(table);
                Value *const A = mvmd_shuffle2(16, T0, T1, IL);
                Value *const B = mvmd_shuffle2(16, T0, T1, IH);
                Value *packed = fwCast(8, hsimd_packl(16, A, B));
                return mCB->CreateAnd(packed, Mask);
            };
#undef ZEXT16L
#undef ZEXT16H

            Value *const L = SelectFromHalfTable(table0, InL);
            assert(L->getType() == vty);
            Value *const H = SelectFromHalfTable(table1, InH);

            assert(H->getType() == vty);
            Value *shuf = mCB->CreateOr(L, H);
            if (mode == ShuffleMode::ZeroOnHighIndexBit) {
                Constant *high_bit = Constant::getIntegerValue(fwTy, APInt::getHighBitsSet(fw, 1));
                Value *enable_shuf = simd_ult(fw, index_vector, getSplat(fieldCount, high_bit));
                return simd_and(shuf, enable_shuf);
            } else if (mode == ShuffleMode::ZeroOnIndexOver) {
                Value *enable_shuf =
                    simd_ult(fw, index_vector, getSplat(fieldCount, ConstantInt::get(fwTy, fieldCount)));
                return simd_and(shuf, enable_shuf);
            }
            return shuf; // if (mode == ShuffleMode::TruncateIndex)
        }
    }
    return IDISA_AVX2_Builder::mvmd_shuffle2_impl(fw, table0, table1, index_vector, mode);
}

Value *IDISA_AVX512F_Builder::mvmd_compress_impl(unsigned fw, Value *a, Value *select_mask) {
    if (getVectorBitWidth(a) == AVX512_width) {
        unsigned fieldCount = getVectorBitWidth(a) / fw;
        Value *mask = mCB->CreateZExtOrTrunc(select_mask, mCB->getIntNTy(fieldCount));
        if (fw == 32) {
            Type *maskTy = FixedVectorType::get(mCB->getInt1Ty(), fieldCount);
            Function *compressFunc = Intrinsic::getOrInsertDeclaration(
                mCB->getModule(), Intrinsic::x86_avx512_mask_compress, fwVectorType(fw));
            return mCB->CreateCall(compressFunc->getFunctionType(), compressFunc,
                                   {fwCast(32, a), fwCast(32, allZeroes()), mCB->CreateBitCast(mask, maskTy)});
        }
        if (fw == 64) {
            Type *maskTy = FixedVectorType::get(mCB->getInt1Ty(), fieldCount);
            Function *compressFunc = Intrinsic::getOrInsertDeclaration(
                mCB->getModule(), Intrinsic::x86_avx512_mask_compress, fwVectorType(fw));
            return mCB->CreateCall(compressFunc->getFunctionType(), compressFunc,
                                   {fwCast(64, a), fwCast(64, allZeroes()), mCB->CreateBitCast(mask, maskTy)});
        }

        if (fw == 8) {
            if (mCB->hasFeature(codegen::Feature::AVX512_VBMI2)) {
                Type *maskTy = FixedVectorType::get(mCB->getInt1Ty(), fieldCount);
                Function *compressFunc = Intrinsic::getOrInsertDeclaration(
                    mCB->getModule(), Intrinsic::x86_avx512_mask_compress, fwVectorType(fw));
                return mCB->CreateCall(compressFunc->getFunctionType(), compressFunc,
                                       {fwCast(8, a), fwCast(8, allZeroes()), mCB->CreateBitCast(mask, maskTy)});
            } else if (mCB->hasFeature(codegen::Feature::AVX512_VBMI) || mCB->hasFeature(codegen::Feature::AVX512_BW)) {

                // Step 1: Initialize indices as 6-bit bixnum in an array of 64-bit integers
                uint64_t indices[6] = {0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC, 0xF0F0F0F0F0F0F0F0,
                                       0xFF00FF00FF00FF00, 0xFFFF0000FFFF0000, 0xFFFFFFFF00000000};

                // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected
                Function *pextFunc = nullptr;
                if (LLVM_LIKELY(fieldCount == 64)) {
                    pextFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_64);
                } else {
                    assert(fieldCount <= 32);
                    pextFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pext_32);
                    if (LLVM_UNLIKELY(fieldCount < 32)) {
                        mask = mCB->CreateZExt(mask, mCB->getInt32Ty());
                    }
                }

                // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

                Value *permute_vec = nullptr;
                const auto m = floor_log2(fieldCount);
                FixedArray<Value *, 2> args;
                args[1] = select_mask;

                IntegerType *intTy = mCB->getIntNTy((fieldCount == 64) ? 64 : 32);

                Type *vTy = fwVectorType(fw);

                for (unsigned i = 0; i < m; ++i) {
                    args[0] = ConstantInt::get(intTy, indices[i]);
                    Value *const expanded = mCB->CreateCall(pextFunc->getFunctionType(), pextFunc, args);
                    Value *byteExpanded = esimd_bitspread(AVX512_width, fw, expanded);
                    if (i == 0) {
                        permute_vec = byteExpanded;
                    } else {
                        Value *shiftedExpand = simd_slli(64, byteExpanded, i);
                        permute_vec = simd_or(permute_vec, shiftedExpand);
                    }
                }

                // // Step 4: Use mvmd_shuffle2 to shuffle using permute_vec
                Constant *zero_vec = ConstantVector::getNullValue(vTy);
                Value *const shuffled = mvmd_shuffle2(fw, a, zero_vec, mCB->CreateBitCast(permute_vec, vTy));
                Value *const count = mCB->CreatePopcount(mCB->CreateZExt(select_mask, intTy));
                assert(count->getType() == intTy);
                // try a bitspread + any? check ASM
                Constant *ALL_ONES = ConstantVector::getAllOnesValue(vTy);
                Value *mask = mCB->CreateNot(mvmd_sll(fw, ALL_ONES, count, false));
                mask = mCB->CreateSelect(mCB->CreateICmpNE(count, ConstantInt::get(intTy, fieldCount)), mask, ALL_ONES);
                assert(shuffled->getType() == mask->getType());
                return mCB->CreateAnd(shuffled, mask);
            }
        }
    }
    return IDISA_AVX2_Builder::mvmd_compress_impl(fw, a, select_mask);
}

#define MVMD_EXPAND_BY_INDUCTIVE_DOUBLING
Value *IDISA_AVX512F_Builder::mvmd_expand_impl(unsigned fw, Value *a, Value *select_mask) {
    if (getVectorBitWidth(a) == AVX512_width) {
        const auto fieldCount = getVectorBitWidth(a) / fw;
        Value *mask = mCB->CreateZExtOrTrunc(select_mask, mCB->getIntNTy(fieldCount));
        bool has_avx_512_mask_expand =
            (fw == 32) || (fw == 64) || (mCB->hasFeature(codegen::Feature::AVX512_VBMI2) && ((fw == 16) | (fw == 8)));
        if (has_avx_512_mask_expand) {
            Type *maskTy = FixedVectorType::get(mCB->getInt1Ty(), fieldCount);
            Function *expandFunc = Intrinsic::getOrInsertDeclaration(
                mCB->getModule(), Intrinsic::x86_avx512_mask_expand, fwVectorType(fw));
            return mCB->CreateCall(expandFunc->getFunctionType(), expandFunc,
                                   {fwCast(fw, a), fwCast(fw, allZeroes()), mCB->CreateBitCast(mask, maskTy)});
        } else if ((fw == 8) || (fw == 16)) {
#ifdef MVMD_EXPAND_BY_INDUCTIVE_DOUBLING
            Value *hi_mask =
                mCB->CreateLShr(mask, Constant::getIntegerValue(mask->getType(), APInt(fieldCount, fieldCount / 2)));
            hi_mask = mCB->CreateTrunc(hi_mask, mCB->getIntNTy(fieldCount / 2));
            Value *lo_mask = mCB->CreateTrunc(mask, mCB->getIntNTy(fieldCount / 2));
            Value *lo_to_hi = mCB->CreatePopcount(mCB->CreateNot(lo_mask));
            Value *lo_vec = esimd_mergel(fw, a, allZeroes());
            Value *hi_vec = esimd_mergeh(fw, mvmd_sll(fw, a, lo_to_hi, true), allZeroes());
            Value *expand_lo = mvmd_expand(fw * 2, lo_vec, lo_mask);
            Value *expand_hi = mvmd_expand(fw * 2, hi_vec, hi_mask);
            Value *packed = bitCast(hsimd_packl(fw * 2, expand_lo, expand_hi));
            return packed;
#else
            uint64_t indices[6] = {0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC, 0xF0F0F0F0F0F0F0F0,
                                   0xFF00FF00FF00FF00, 0xFFFF0000FFFF0000, 0xFFFFFFFF00000000};

            Function *pdepFunc = nullptr;
            if (LLVM_LIKELY(fieldCount == 64)) {
                pdepFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_64);
            } else {
                assert(fieldCount <= 32);
                pdepFunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_bmi_pdep_32);
                if (LLVM_UNLIKELY(fieldCount < 32)) {
                    mask = mCB->CreateZExt(mask, mCB->getInt32Ty());
                }
            }

            // Step 2: Use PEXT instruction to select only the bixnum values for the bytes to be selected

            Value *permute_vec = nullptr;
            const auto m = floor_log2(fieldCount);
            FixedArray<Value *, 2> args;
            args[1] = mask;
            const auto popFW = (fieldCount == 64) ? 64U : 32U;
            for (unsigned i = 0; i < m; ++i) {
                args[0] = getIntN(popFW, indices[i]);
                Value *const expanded = mCB->CreateCall(pdepFunc->getFunctionType(), pdepFunc, args);
                assert(expanded->getType()->getIntegerBitWidth() == popFW);
                Value *byteExpanded = esimd_bitspread(AVX512_width, fw, expanded);
                if (i == 0) {
                    permute_vec = byteExpanded;
                } else {
                    Value *shiftedExpand = simd_slli(64, byteExpanded, i);
                    permute_vec = simd_or(permute_vec, shiftedExpand);
                }
            }

            a = fwCast(fw, a);
            permute_vec = fwCast(fw, permute_vec);

            // // Step 4: Use mvmd_shuffle2 to shuffle using permute_vec
            Constant *zero_vec = ConstantVector::getNullValue(a->getType());
            Value *const shuffled = mvmd_shuffle2(fw, a, zero_vec, permute_vec);
            return mCB->CreateAnd(shuffled, simd_any(fw, esimd_bitspread(AVX512_width, fw, mask)));
#endif
        }
    }
    return IDISA_AVX2_Builder::mvmd_expand_impl(fw, a, select_mask);
}

Value *IDISA_AVX512F_Builder::mvmd_slli_impl(unsigned fw, Value *a, unsigned shift) {
    if (getVectorBitWidth(a) == AVX512_width) {
        if (shift == 0)
            return a;
        if (fw > 32) {
            return fwCast(fw, mvmd_slli(32, a, shift * (fw / 32)));
        } else if (((shift % 2) == 0) && (fw < 32)) {
            return fwCast(fw, mvmd_slli(2 * fw, a, shift / 2));
        }
        if ((fw == 32) || (mCB->hasFeature(codegen::Feature::AVX512_BW) && (fw == 16))) {
            return mvmd_dslli(fw, a, allZeroes(), shift);
        } else {
            unsigned field32_shift = (shift * fw) / 32;
            unsigned bit_shift = (shift * fw) % 32;
            Value *const L = simd_slli(32, mvmd_slli(32, a, field32_shift), bit_shift);
            Value *const R = simd_srli(32, mvmd_slli(32, a, field32_shift + 1), 32 - bit_shift);
            assert(L->getType() == R->getType());
            return fwCast(fw, mCB->CreateOr(L, R));
        }
    }
    return IDISA_AVX2_Builder::mvmd_slli_impl(fw, a, shift);
}

Value *IDISA_AVX512F_Builder::mvmd_dslli_impl(unsigned fw, Value *a, Value *b, unsigned shift) {
    if (getVectorBitWidth(a) == AVX512_width) {
        if (shift == 0)
            return a;
        if (fw > 32) {
            return fwCast(fw, mvmd_dslli(32, a, b, shift * (fw / 32)));
        } else if (((shift % 2) == 0) && (fw < 32)) {
            return fwCast(fw, mvmd_dslli(2 * fw, a, b, shift / 2));
        }
        const unsigned fieldCount = AVX512_width / fw;
        if ((fw == 32) || (mCB->hasFeature(codegen::Feature::AVX512_BW) && (fw == 16))) {
            // llvm::errs() << " fw = " << fw << ", shift = " << shift << "\n";
            Type *fwTy = mCB->getIntNTy(fw);
            SmallVector<Constant *, 16> indices(fieldCount);
            for (unsigned i = 0; i < fieldCount; i++) {
                indices[i] = ConstantInt::get(fwTy, i + fieldCount - shift);
            }
            return mvmd_shuffle2(fw, fwCast(fw, b), fwCast(fw, a), ConstantVector::get(indices));
        } else {
            unsigned field32_shift = (shift * fw) / 32;
            unsigned bit_shift = (shift * fw) % 32;
            Value *const L = simd_slli(32, mvmd_dslli(32, a, b, field32_shift), bit_shift);
            Value *const R = simd_srli(32, mvmd_dslli(32, a, b, field32_shift + 1), 32 - bit_shift);
            assert(L->getType() == R->getType());
            return fwCast(fw, mCB->CreateOr(L, R));
        }
    }
    return IDISA_AVX2_Builder::mvmd_dslli_impl(fw, a, b, shift);
}

Value *IDISA_AVX512F_Builder::simd_popcount_impl(unsigned fw, Value *a) {
    if (getVectorBitWidth(a) == AVX512_width) {
        if (fw == 512) {
            Constant *zero16xi8 = Constant::getNullValue(FixedVectorType::get(mCB->getInt8Ty(), 16));
            Constant *zeroInt32 = Constant::getNullValue(mCB->getInt32Ty());
            Value *c = simd_popcount(64, a);
            //  Should probably use _mm512_reduce_add_epi64, but not found in LLVM 3.8
            Function *pack64_8_func =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx512_mask_pmov_qb_512);
            // popcounts of 64 bit fields will always fit in 8 bit fields.
            // We don't need the masked version of this, but the unmasked intrinsic was not found.
            c = mCB->CreateCall(pack64_8_func->getFunctionType(), pack64_8_func,
                                {c, zero16xi8, Constant::getAllOnesValue(mCB->getInt8Ty())});
            Function *horizSADfunc = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_sse2_psad_bw);
            c = mCB->CreateCall(horizSADfunc->getFunctionType(), horizSADfunc, {c, zero16xi8});
            return mCB->CreateInsertElement(allZeroes(), mCB->CreateExtractElement(c, zeroInt32), zeroInt32);
        }
        if (mCB->hasFeature(codegen::Feature::AVX512_VPOPCNTDQ) && (fw == 32 || fw == 64)) {
            // llvm should use vpopcntd or vpopcntq instructions
            return mCB->CreatePopcount(fwCast(fw, a));
        }
        if (mCB->hasFeature(codegen::Feature::AVX512_BW) && (fw == 64)) {
            Function *horizSADfunc =
                Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx512_psad_bw_512);
            return mCB->CreateCall(horizSADfunc->getFunctionType(), horizSADfunc,
                                   {fwCast(8, simd_popcount(8, a)), fwCast(8, allZeroes())});
        }
        // https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation
        if (fw == 64) {
            Constant *m1Arr[8];
            Constant *m1;
            for (unsigned int i = 0; i < 8; i++) {
                m1Arr[i] = mCB->getInt64(0x5555555555555555);
            }
            m1 = ConstantVector::get({m1Arr, 8});

            Constant *m2Arr[8];
            Constant *m2;
            for (unsigned int i = 0; i < 8; i++) {
                m2Arr[i] = mCB->getInt64(0x3333333333333333);
            }
            m2 = ConstantVector::get({m2Arr, 8});

            Constant *m4Arr[8];
            Constant *m4;
            for (unsigned int i = 0; i < 8; i++) {
                m4Arr[i] = mCB->getInt64(0x0f0f0f0f0f0f0f0f);
            }
            m4 = ConstantVector::get({m4Arr, 8});

            Constant *h01Arr[8];
            Constant *h01;
            for (unsigned int i = 0; i < 8; i++) {
                h01Arr[i] = mCB->getInt64(0x0101010101010101);
            }
            h01 = ConstantVector::get({h01Arr, 8});

            a = simd_sub(fw, a, simd_and(simd_srli(fw, a, 1), m1));
            a = simd_add(fw, simd_and(a, m2), simd_and(simd_srli(fw, a, 2), m2));
            a = simd_and(simd_add(fw, a, simd_srli(fw, a, 4)), m4);
            return simd_srli(fw, simd_mult(fw, a, h01), 56);
        }
    }
    return IDISA_AVX2_Builder::simd_popcount_impl(fw, a);
}

Value *IDISA_AVX512F_Builder::hsimd_signmask_impl(unsigned fw, Value *a) {
    // IDISA_Builder::hsimd_signmask outperforms IDISA_AVX2_Builder::hsimd_signmask
    // when run with BlockSize=512
    return IDISA_AVX2_Builder::hsimd_signmask_impl(fw, a);
}

Value *IDISA_AVX512F_Builder::esimd_mergeh_impl(unsigned fw, Value *a, Value *b) {
    if (getVectorBitWidth(a) == AVX512_width) {
        if (mCB->hasFeature(codegen::Feature::AVX512_BW) && ((fw == 1) || (fw == 2))) {
            // Bit interleave using shuffle.
            // Make a shuffle table that translates the lower 4 bits of each byte in
            // order to spread out the bits: xxxxdcba => .d.c.b.a
            // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
            Constant *interleave_table = bit_interleave_byteshuffle_table(fw);
            // Merge the bytes.
            Value *byte_merge = esimd_mergeh(8, a, b);
            Function *shufFn = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx512_pshuf_b_512);
            Value *low_bits = mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                              {interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8)))});
            Value *high_bits = simd_slli(16,
                                         mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                                         {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}),
                                         fw);
            Value *lo_move_back = simd_srli(16, low_bits, 8 - fw);
            Value *hi_move_fwd = simd_slli(16, high_bits, 8 - fw);
            return simd_or(simd_if(1, simd_himask(16), high_bits, low_bits), simd_or(lo_move_back, hi_move_fwd));
        }
    }
    // Otherwise use default AVX2 logic.
    return IDISA_AVX2_Builder::esimd_mergeh_impl(fw, a, b);
}

Value *IDISA_AVX512F_Builder::esimd_mergel_impl(unsigned fw, Value *a, Value *b) {
    if ((getVectorBitWidth(a) == AVX512_width) && mCB->hasFeature(codegen::Feature::AVX512_BW) &&
        ((fw == 1) || (fw == 2))) {
        // Bit interleave using shuffle.
        // Make a shuffle table that translates the lower 4 bits of each byte in
        // order to spread out the bits: xxxxdcba => .d.c.b.a
        // We use two copies of the table for the AVX2 _mm256_shuffle_epi8
        Constant *interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value *byte_merge = esimd_mergel(8, a, b);
        Function *shufFn = Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx512_pshuf_b_512);
        Value *low_bits = mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                          {interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8)))});
        Value *high_bits = simd_slli(16,
                                     mCB->CreateCall(shufFn->getFunctionType(), shufFn,
                                                     {interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))}),
                                     fw);
        Value *lo_move_back = simd_srli(16, low_bits, 8 - fw);
        Value *hi_move_fwd = simd_slli(16, high_bits, 8 - fw);
        return simd_or(simd_if(1, simd_himask(16), high_bits, low_bits), simd_or(lo_move_back, hi_move_fwd));
    }
    // Otherwise use default AVX2 logic.
    return IDISA_AVX2_Builder::esimd_mergel_impl(fw, a, b);
}

Value *IDISA_AVX512F_Builder::simd_if_impl(unsigned fw, Value *cond, Value *a, Value *b) {
    if ((getVectorBitWidth(a) == AVX512_width) && (fw == 1)) {
        // Form the 8-bit table for simd-if based on the bitwise values from cond, a and b.
        //   (cond, a, b) =  (111), (110), (101), (100), (011), (010), (001), (000)
        // if(cond, a, b) =    1      1      0      0      1      0      1      0    = 0xCA
        return simd_ternary(0xCA, bitCast(cond), bitCast(a), bitCast(b));
    }
    return IDISA_AVX2_Builder::simd_if_impl(fw, cond, a, b);
}

Value *IDISA_AVX512F_Builder::simd_ternary_impl(unsigned char mask, Value *a, Value *b, Value *c) {
    if (getVectorBitWidth(a) == AVX512_width) {
        assert(a->getType() == b->getType());
        assert(b->getType() == c->getType());

        if (mask == 0) {
            return allZeroes();
        }
        if (mask == 0xFF) {
            return allOnes();
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

        Constant *simd_mask = mCB->getInt32(mask);
        Function *ternLogicFn =
            Intrinsic::getOrInsertDeclaration(mCB->getModule(), Intrinsic::x86_avx512_pternlog_d_512);
        Value *args[4] = {fwCast(32, a), fwCast(32, b), fwCast(32, c), simd_mask};
        return bitCast(mCB->CreateCall(ternLogicFn->getFunctionType(), ternLogicFn, args));
    }
    return IDISA_AVX2_Builder::simd_ternary_impl(mask, a, b, c);
}

std::pair<Value *, Value *> IDISA_AVX512F_Builder::bitblock_advance_impl(Value *a, Value *shiftin, unsigned shift) {
    if (getVectorBitWidth(a) == AVX512_width) {
        if (shift == 1 && shiftin->getType() == mCB->getInt8Ty()) {
            const uint32_t fw = 64;
            Value *const ci_mask = mCB->CreateBitCast(shiftin, FixedVectorType::get(mCB->getInt1Ty(), 8));
            Value *const v8xi64_1 = simd_fill(fw, mCB->getInt64(0x8000000000000000));
            Value *const ecarry_in = mCB->CreateSelect(
                ci_mask, v8xi64_1, Constant::getNullValue(FixedVectorType::get(mCB->getInt64Ty(), 8)));
            Value *const a1 = mvmd_dslli(fw, a, ecarry_in, shift);
            Value *const result = simd_or(mCB->CreateLShr(a1, fw - shift), mCB->CreateShl(fwCast(fw, a), shift));

            std::vector<Constant *> v(8, mCB->getInt64((uint64_t)-1));
            v[7] = mCB->getInt64(0x7fffffffffffffff);
            Value *const v8xi64_cout_mask = ConstantVector::get(ArrayRef<Constant *>(v));
            Value *shiftout = mCB->CreateICmpUGT(a, v8xi64_cout_mask);
            shiftout = mCB->CreateBitCast(shiftout, mCB->getInt8Ty());
            return std::make_pair(shiftout, result);
        }
    }
    return IDISA_AVX2_Builder::bitblock_advance_impl(a, shiftin, shift);
}

} // namespace IDISA
