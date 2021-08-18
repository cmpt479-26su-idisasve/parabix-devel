#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 128 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

/* Creates a call to neon_vldq to shift 4 vector */
// Value * IDISA_ARM_Builder::neon_vld1x4() {
//   // create a function * to the vld1x4
//   // Function * shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld1, VectorType::get(getInt32Ty(), 4));
//   // Function * shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld4, {getInt32Ty()});
//   // create a static vector of {0, 1, 2, 3} that represents the shift amounts
//   std::vector<Constant *> shuffle_amount = {
//     ConstantInt::get(getInt32Ty(), 0),
//     ConstantInt::get(getInt32Ty(), 1),
//     ConstantInt::get(getInt32Ty(), 2),
//     ConstantInt::get(getInt32Ty(), 3)
//   };
//   Constant * llshuffle_amount = ConstantVector::get(shuffle_amount);
//   // create a function call
//   // return CreateCall(
//   //   shiftl_f32func->getFunctionType(),
//   //   shiftl_f32func,
//   //   llshuffle_amount
//   // );
//   return llshuffle_amount;
// }

/* shift lefts everything */
// Value * IDISA_ARM_Builder::neon_shlq(Value * a, Value * b) {
//   Function * shiftl_u_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vshiftu);
//   return CreateCall(shiftl_u_f32func->getFunctionType(), shiftl_u_f32func, {a, b});
// }

// Value * IDISA_ARM_Builder::hsimd_signmask(unsigned fw, Value * a) {
//    /*
//    SIMDE implementation
//    static const int32_t shift_amount[] = {0, 1, 2, 3};
//    const int32x4_t shift = vld1q_s32(shift_amount);
//    uint32x4_t tmp = vshrq_n_u32(a, 31);
//    return HEDLEY_STATIC_CAST(LLVM::Value *, vaddvq_u32(vshlq_u32(tmp, shift)));
//    */

//   if (getVectorBitWidth(a) == ARM_width) {
//     if (fw == 32) {
//       // CallPrintRegister("a", fwCast(fw, a));
//       /* shift operation */
//       auto shift = neon_vld1x4();
//       CallPrintRegister("shift", shift);
//       llvm::errs() << "shift\n";
//       // Value * temp = neon_shrq(a);
//       Value * temp = CreateLShr(fwCast(fw, a), 31);
//       llvm::errs() << "temp\n";
//       // Value * shift_left_a = neon_shlq(temp, shift);
//       Value * shift_left_a = CreateShl(temp, shift);
//       llvm::errs() << "shift_left_a\n";

//       /* finally add it to the vector */
//       // Function * add_32func = Intrinsic::getDeclaration(
//       //   getModule(),
//       //   Intrinsic::arm_neon_vqaddu,
//       //   VectorType::get(getInt32Ty(), getVectorBitWidth(shift_left_a))
//       // );
//       // Function * add_32func = CreateAddReduce(fwCast(fw, shift_left_a));
//       // printf("%p\n", add_32func);
//       // return CreateCall(add_32func->getFunctionType(), add_32func, shift_left_a);

//       return CreateAddReduce(shift_left_a);
//     }
//   }
// }

Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {
  if (mBitBlockWidth == 128 && fw > 8) {
    // Create a table for shuffling with smaller field widths.
    const unsigned fieldCount = mBitBlockWidth/fw;
    Constant * idxMask = ConstantVector::getSplat(fieldCount, ConstantInt::get(getIntNTy(fw), fieldCount-1));
    Value * idx = simd_and(index_vector, idxMask);
    unsigned half_fw = fw/2;
    unsigned field_count = mBitBlockWidth/half_fw;
    // Build a ConstantVector of alternating 0 and 1 values.
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned int i = 0; i < field_count; i++) {
      Idxs[i] = ConstantInt::get(getIntNTy(fw/2), i & 1);
    }
    Constant * splat01 = ConstantVector::get(Idxs);
    
    Value * half_fw_indexes = simd_or(idx, mvmd_slli(half_fw, idx, 1));
    half_fw_indexes = simd_add(fw, simd_add(fw, half_fw_indexes, half_fw_indexes), splat01);
    Value * rslt = mvmd_shuffle(half_fw, data_table, half_fw_indexes);
    return rslt;
  }
  if (mBitBlockWidth == 128 && fw == 8) {
    Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_tbl1, VectorType::get(getInt8Ty(), 16));
    return fwCast(8, CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, data_table), fwCast(8, simd_select_lo(fw, index_vector))}));
  }
  return IDISA_Builder::mvmd_shuffle(fw, data_table, index_vector);
}

Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {
  // SSE
  if ((mBitBlockWidth == 128) && (fw == 64)) {
    Constant * keep[2] = {ConstantInt::get(getInt64Ty(), 1), ConstantInt::get(getInt64Ty(), 3)};
    Constant * keep_mask = ConstantVector::get({keep, 2});
    Constant * shift[2] = {ConstantInt::get(getInt64Ty(), 2), ConstantInt::get(getInt64Ty(), 0)};
    Constant * shifted_mask = ConstantVector::get({shift, 2});
    Value * a_srli1 = mvmd_srli(64, a, 1);
    Value * bdcst = simd_fill(64, CreateZExt(selector, getInt64Ty()));
    Value * kept = simd_and(simd_eq(64, simd_and(keep_mask, bdcst), keep_mask), a);
    Value * shifted = simd_and(a_srli1, simd_eq(64, shifted_mask, bdcst));
    return simd_or(kept, shifted);
    }
    if ((mBitBlockWidth == 128) && (fw == 32)) {
      Value * bdcst = simd_fill(32, CreateZExtOrTrunc(selector, getInt32Ty()));
      Constant * fieldBit[4] =
      {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 2),
        ConstantInt::get(getInt32Ty(), 4), ConstantInt::get(getInt32Ty(), 8)};
      Constant * fieldMask = ConstantVector::get({fieldBit, 4});
      Value * a_selected = simd_and(simd_eq(32, fieldMask, simd_and(fieldMask, bdcst)), a);
      Constant * rotateInwards[4] =
      {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 0),
        ConstantInt::get(getInt32Ty(), 3), ConstantInt::get(getInt32Ty(), 2)};
      Constant * rotateVector = ConstantVector::get({rotateInwards, 4});
      Value * rotated = CreateShuffleVector(fwCast(32, a_selected), UndefValue::get(fwVectorType(fw)), rotateVector);
      Constant * rotate_bit[2] = {ConstantInt::get(getInt64Ty(), 2), ConstantInt::get(getInt64Ty(), 4)};
      Constant * rotate_mask = ConstantVector::get({rotate_bit, 2});
      Value * rotateControl = simd_eq(64, fwCast(64, simd_and(bdcst, rotate_mask)), allZeroes());
      Value * centralResult = simd_if(1, rotateControl, rotated, a_selected);
      Value * delete_marks_lo = CreateAnd(CreateNot(selector), ConstantInt::get(selector->getType(), 3));
      Value * delCount_lo = CreateSub(delete_marks_lo, CreateLShr(delete_marks_lo, 1));
      return mvmd_srl(32, centralResult, delCount_lo, true);
    }
    return IDISA_Builder::mvmd_compress(fw, a, selector);
}

// Value * IDISA_ARM_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
//   // SSE2
//   if ((fw == 16) && (getVectorBitWidth(a) == ARM_width)) {
//     Value * mask = simd_lomask(16);
//     return hsimd_packus(fw, fwCast(16, simd_and(a, mask)), fwCast(16, simd_and(b, mask)));
//   }
//   // Otherwise use default logic.
//   return IDISA_Builder::hsimd_packl(fw, a, b);
// }

Value * IDISA_ARM_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
  if ((fw == 16) && (getVectorBitWidth(a) == ARM_width)) {
    Function * vqmovun_s16_func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_uqxtn, VectorType::get(getInt8Ty(), 8));
    Value * sat_a = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, simd_srli(16, a, 8));
    Value * sat_b = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, simd_srli(16, b, 8));
    return fwCast(8, CreateDoubleVector(sat_a, sat_b));
    // return CreateCall(packuswb_func->getFunctionType(), packuswb_func, {simd_srli(16, a, 8), simd_srli(16, b, 8)});
    // return IDISA_Builder::hsimd_packh(fw, a, b);
  }
  // Otherwise use default logic.
  return IDISA_Builder::hsimd_packh(fw, a, b);
}

// Value * IDISA_ARM_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {

// }

// // full shift producing {shiftout, shifted}
// std::pair<Value *, Value *> IDISA_ARM_Builder::bitblock_advance(Value * a, Value * shiftin, unsigned shift) {

// }

// Value * IDISA_Builder::hsimd_partial_sum(unsigned fw, Value * a)

Value * IDISA_ARM_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
  if ((fw == 1) || (fw == 2)) {
    Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
    // Merge the bytes.
    Value * byte_merge = esimd_mergeh(8, a, b);
    Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
    // Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_select_lo(8, byte_merge)));
    Value * high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
    // For each 16-bit field, interleave the low bits of the two bytes.
    low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
    // For each 16-bit field, interleave the high bits of the two bytes.
    high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
    return simd_or(low_bits, high_bits);
  }
  // Otherwise use default SSE logic.
  return IDISA_Builder::esimd_mergeh(fw, a, b);
}

Value * IDISA_ARM_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
  if ((fw == 1) || (fw == 2)) {
    Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
    // Merge the bytes.
    Value * byte_merge = esimd_mergel(8, a, b);
    Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
    // Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_select_lo(8, byte_merge)));
    Value * high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
    // For each 16-bit field, interleave the low bits of the two bytes.
    low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
    // For each 16-bit field, interleave the high bits of the two bytes.
    high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
    return simd_or(low_bits, high_bits);
  }
  // Otherwise use default SSE2 logic.
  return IDISA_Builder::esimd_mergel(fw, a, b);
}

}
