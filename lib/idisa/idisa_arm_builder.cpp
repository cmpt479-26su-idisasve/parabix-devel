#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

#include <stdio.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 64 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

/* Creates a call to neon_vldq to shift 4 vector */
Value * IDISA_ARM_Builder::neon_vld1x4() {
  // create a function * to the vld1x4
  // Function * shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld1, VectorType::get(getInt32Ty(), 4));
  Function * shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld4, {getInt32Ty()});
  // create a static vector of {0, 1, 2, 3} that represents the shift amounts
  return CreateCall(
    shiftl_f32func->getFunctionType(),
    shiftl_f32func,
    {
      ConstantInt::get(getInt32Ty(), 0),
      ConstantInt::get(getInt32Ty(), 1),
      ConstantInt::get(getInt32Ty(), 2),
      ConstantInt::get(getInt32Ty(), 3)
    }
  );
}

/* shift lefts everything */
Value * IDISA_ARM_Builder::neon_shlq(Value * a, Value * b) {
  Function * shiftl_u_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vshiftu);
  return CreateCall(shiftl_u_f32func->getFunctionType(), shiftl_u_f32func, {a, b});
}

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
      /* shift operation */
      Value * shift = neon_vld1x4();
      printf("shift %p\n", shift);
      // Value * temp = neon_shrq(a);
      Value * temp = CreateLShr(fwCast(fw, a), 31);
      printf("temp %p\n", temp);
      // Value * shift_left_a = neon_shlq(temp, shift);
      Value * shift_left_a = CreateShl(temp, shift);
      printf("shift_left_a %p\n", shift_left_a);

      return CreateAddReduce(shift_left_a);
    }
  }
}

//Value * IDISA_ARM_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {    

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
