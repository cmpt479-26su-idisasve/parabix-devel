#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

// #include <stdio.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 64 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

/* Creates a call to neon_vldq to shift 4 vector */
Value * IDISA_ARM_Builder::neon_vld1x4() {
  // create a function * to the vld1x4
  // Function * shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld1, VectorType::get(getInt32Ty(), 4));
  // Function * shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld4, {getInt32Ty()});
  // create a static vector of {0, 1, 2, 3} that represents the shift amounts
  std::vector<Constant *> shuffle_amount = {
    ConstantInt::get(getInt32Ty(), 0),
    ConstantInt::get(getInt32Ty(), 1),
    ConstantInt::get(getInt32Ty(), 2),
    ConstantInt::get(getInt32Ty(), 3)
  };
  Constant * llshuffle_amount = ConstantVector::get(shuffle_amount);
  // create a function call
  // return CreateCall(
  //   shiftl_f32func->getFunctionType(),
  //   shiftl_f32func,
  //   llshuffle_amount
  // );
  return llshuffle_amount;
}

/* shift lefts everything */
// Value * IDISA_ARM_Builder::neon_shlq(Value * a, Value * b) {
//   Function * shiftl_u_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vshiftu);
//   return CreateCall(shiftl_u_f32func->getFunctionType(), shiftl_u_f32func, {a, b});
// }

Value * IDISA_ARM_Builder::hsimd_signmask(unsigned fw, Value * a) {
   /*
   SIMDE implementation
   static const int32_t shift_amount[] = {0, 1, 2, 3};
   const int32x4_t shift = vld1q_s32(shift_amount);
   uint32x4_t tmp = vshrq_n_u32(a, 31);
   return HEDLEY_STATIC_CAST(LLVM::Value *, vaddvq_u32(vshlq_u32(tmp, shift)));
   */

  if (getVectorBitWidth(a) == ARM_width) {
    if (fw == 32) {
      // CallPrintRegister("a", fwCast(fw, a));
      /* shift operation */
      auto shift = neon_vld1x4();
      CallPrintRegister("shift", shift);
      llvm::errs() << "shift\n";
      // Value * temp = neon_shrq(a);
      Value * temp = CreateLShr(fwCast(fw, a), 31);
      llvm::errs() << "temp\n";
      // Value * shift_left_a = neon_shlq(temp, shift);
      Value * shift_left_a = CreateShl(temp, shift);
      llvm::errs() << "shift_left_a\n";

      /* finally add it to the vector */
      // Function * add_32func = Intrinsic::getDeclaration(
      //   getModule(),
      //   Intrinsic::arm_neon_vqaddu,
      //   VectorType::get(getInt32Ty(), getVectorBitWidth(shift_left_a))
      // );
      // Function * add_32func = CreateAddReduce(fwCast(fw, shift_left_a));
      // printf("%p\n", add_32func);
      // return CreateCall(add_32func->getFunctionType(), add_32func, shift_left_a);

      return CreateAddReduce(shift_left_a);
    }
  }
}

// SSE2
Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {
  if ((mBitBlockWidth == 128) && (fw == 64)) {
    // First create a vector with exchanged values of the 2 fields.
    Constant * idx[2] = {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 0)};
    Value * exchanged = CreateShuffleVector(data_table, UndefValue::get(fwVectorType(fw)), ConstantVector::get({idx, 2}));
    // bits that change if the value in a needs to be exchanged.
    Value * changed = simd_xor(data_table, exchanged);
    // Now create a mask to select between original and exchanged values.
    Constant * xchg[2] = {ConstantInt::get(getInt64Ty(), 1), ConstantInt::get(getInt64Ty(), 0)};
    Value * xchg_vec = ConstantVector::get({xchg, 2});
    Constant * oneSplat = ConstantVector::getSplat(2, ConstantInt::get(getInt64Ty(), 1));
    Value * exchange_mask = simd_eq(fw, simd_and(index_vector, oneSplat), xchg_vec);
    Value * rslt = simd_xor(simd_and(changed, exchange_mask), data_table);
    return rslt;
  }
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
    CallPrintRegister("data_table", data_table);
    // Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_ssse3_pshuf_b_128);
    Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vtbl2); // arm_neon_vtbl1  arm_neon_vtbx1
    // Value * dt_cast = fwCast(8, data_table);
    Value * loBits = CreateHalfVectorLow(data_table);
    Value * highBits = CreateHalfVectorHigh(data_table);
    Value * loIdx = CreateHalfVectorLow(index_vector);
    Value * highIdx = CreateHalfVectorHigh(index_vector);

    Value * lowShuffle = CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, loBits), fwCast(8, highBits), fwCast(8, simd_select_lo(fw, loIdx))});
    // loBits->print(llvm::errs());
    // llvm::errs() << "\n";
    Value * highShuffle = CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, loBits), fwCast(8, highBits), fwCast(8, simd_select_hi(fw, highIdx))});
    // shuf8Func->getType()->print(llvm::errs());
    // llvm::errs() << "shuf8Func declared\n";
    // return CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, data_table), fwCast(8, simd_and(index_vector, simd_lomask(8))), fwCast(8, simd_and(index_vector, simd_lomask(8)))});
    // Value * res = fwCast(8, CreateDoubleVector(fwCast(8, lowShuffle), fwCast(8, highShuffle)));
    // res->print(llvm::errs());
    // llvm::errs() << "\n";
    // return res;
    return fwCast(8, CreateDoubleVector(lowShuffle, highShuffle));
  }
  return IDISA_Builder::mvmd_shuffle(fw, data_table, index_vector);
}

// Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {

     //const int16x8_t t1 = vcombine_s16(a_.neon_i16, b_.neon_i16);
      /* Set elements which are < 0 to 0 */
     // const int16x8_t t2 = vandq_s16(t1, vreinterpretq_s16_u16(vcgezq_s16(t1)));
      /* Vector with all s16 elements set to UINT8_MAX */
     // const int16x8_t vmax = vmovq_n_s16(HEDLEY_STATIC_CAST(int16_t, UINT8_MAX));
      /* Elements which are within the acceptable range */
     // const int16x8_t le_max = vandq_s16(t2, vreinterpretq_s16_u16(vcleq_s16(t2, vmax)));
     // const int16x8_t gt_max = vandq_s16(vmax, vreinterpretq_s16_u16(vcgtq_s16(t2, vmax)));
      /* Final values as 16-bit integers */
     // const int16x8_t values = vorrq_s16(le_max, gt_max);
     // r_.neon_u8 = vmovn_u16(vreinterpretq_u16_s16(values));

//}

// Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {

// }

// Value * IDISA_ARM_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {

// }

// Value * IDISA_ARM_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {

// }

// // full shift producing {shiftout, shifted}
// std::pair<Value *, Value *> IDISA_ARM_Builder::bitblock_advance(Value * a, Value * shiftin, unsigned shift) {

// }

// llvm::Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {

// }

Value * IDISA_ARM_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
    if ((fw == 1) || (fw == 2)) {
        Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
        // Merge the bytes.
        Value * byte_merge = esimd_mergeh(8, a, b);
        Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
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
