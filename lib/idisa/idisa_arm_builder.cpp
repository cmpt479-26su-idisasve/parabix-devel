#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

// #include <stdio.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 64 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

/* mvmd_shuffle arm */
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
    Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_tbl1, VectorType::get(getInt8Ty(), 16)); // arm_neon_vtbl1  arm_neon_vtbx1 arm_neon_vtbl2 aarch64_neon_tbl1
    return fwCast(8, CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, data_table), fwCast(8, simd_select_lo(fw, index_vector))}));
  }
  return IDISA_Builder::mvmd_shuffle(fw, data_table, index_vector);
}

// Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {
//}

// Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {

// }

// Value * IDISA_ARM_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {

// }

// Value * IDISA_ARM_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {

// }

// llvm::Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {

// }

/* merge_h arm */
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

/* merge_l arm */
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
