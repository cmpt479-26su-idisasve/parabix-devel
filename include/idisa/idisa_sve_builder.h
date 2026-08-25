#pragma once

#include <idisa/idisa_neon_builder.h>

namespace IDISA {

constexpr unsigned SVE_min_width = 128;

class IDISA_SVE_Builder : public IDISA_Generic_Builder {
  public:
    explicit IDISA_SVE_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth);

    std::string getBuilderCacheName() override;

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
    llvm::Value *simd_popcount_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_bitreverse_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                              ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;

  private:
    template <class F> llvm::Value *encapsulateScalableUnary(unsigned fw, llvm::Value *param, F &&createOp);
    template <class F>
    llvm::Value *encapsulateScalableBinary(unsigned fw, llvm::Value *param1, llvm::Value *param2, F &&createOp);

    IDISA_Neon_Builder mNeonB;
};

} // namespace IDISA
