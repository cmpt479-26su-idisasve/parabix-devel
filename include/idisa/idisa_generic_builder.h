#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <bitset>
#include <idisa/CBuilder.h>
#include <idisa/idisa_builder.h>
#include <llvm/IR/DerivedTypes.h>
#include <string>
#include <toolchain/toolchain.h>

namespace IDISA {

class IDISA_Generic_Builder : public IDISA_Builder {
  protected:
    llvm::Value *simd_fill_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_fill_impl(unsigned vector_width, unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_add_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_sub_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_mult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_eq_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ne_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_gt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_lt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_le_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ugt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ule_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_uge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_max_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_umax_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_min_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_umin_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_if_impl(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ternary_impl(unsigned char mask, llvm::Value *bit_2, llvm::Value *bit_1,
                                   llvm::Value *bit_0) override;
    llvm::Value *simd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) override;
    llvm::Value *simd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) override;
    llvm::Value *simd_srai_impl(unsigned fw, llvm::Value *a, unsigned shift) override;
    llvm::Value *simd_sllv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) override;
    llvm::Value *simd_srlv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) override;
    llvm::Value *simd_rotl_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) override;
    llvm::Value *simd_rotr_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) override;
    std::vector<llvm::Value *> simd_pext_impl(unsigned fw, std::vector<llvm::Value *>,
                                              llvm::Value *extract_mask) override;
    llvm::Value *simd_pdep_impl(unsigned fw, llvm::Value *v, llvm::Value *deposit_mask) override;
    llvm::Value *simd_any_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_popcount_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *hsimd_partial_sum_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_cttz_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_bitreverse_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_bitspread_impl(unsigned vec_width, unsigned fw, llvm::Value *bitmask) override;
    llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packss_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packh_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *mvmd_extract_impl(unsigned fw, llvm::Value *a, unsigned fieldIndex) override;
    llvm::Value *mvmd_insert_impl(unsigned fw, llvm::Value *blk, llvm::Value *elt, unsigned fieldIndex) override;
    llvm::Value *mvmd_sll_impl(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe = false) override;
    llvm::Value *mvmd_srl_impl(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe = false) override;
    llvm::Value *mvmd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) override;
    llvm::Value *mvmd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) override;
    llvm::Value *mvmd_dslli_impl(unsigned fw, llvm::Value *a, llvm::Value *b, unsigned shift) override;
    llvm::Value *mvmd_dsll_impl(unsigned fw, llvm::Value *a, llvm::Value *b, llvm::Value *shift) override;
    llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                   ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                                    ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *bitblock_any_impl(llvm::Value *a) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_add_with_carry_impl(llvm::Value *a, llvm::Value *b,
                                                                         llvm::Value *carryin) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_subtract_with_borrow_impl(llvm::Value *a, llvm::Value *b,
                                                                               llvm::Value *borrowin) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance_impl(llvm::Value *a, llvm::Value *shiftin,
                                                                  unsigned shift) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_indexed_advance_impl(llvm::Value *a, llvm::Value *index_strm,
                                                                          llvm::Value *shiftin,
                                                                          unsigned shift) override;
    llvm::Value *bitblock_mask_from_impl(llvm::Value *const position, const bool safe = false) override;
    llvm::Value *bitblock_mask_to_impl(llvm::Value *const position, const bool safe = false) override;
    llvm::Value *bitblock_set_bit_impl(llvm::Value *const position, const bool safe = false) override;

  protected:
    const unsigned mNativeBitBlockWidth;
    const unsigned mMaxNativeSimdShift;
    const unsigned mMinNativeSimdShift;

    IDISA_Generic_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth, unsigned nativeVectorWidth,
                          unsigned maxShiftFw = 64, unsigned minShiftFw = 16)
        : IDISA_Builder(cb, vectorWidth, laneWidth), mNativeBitBlockWidth(nativeVectorWidth),
          mMaxNativeSimdShift(maxShiftFw), mMinNativeSimdShift(minShiftFw) {}
};

} // namespace IDISA
