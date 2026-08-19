#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_sse_builder.h>
#include <toolchain/toolchain.h>

namespace IDISA {

constexpr unsigned AVX_width = 256;

constexpr unsigned AVX512_width = 512;

class IDISA_AVX_Builder : public IDISA_SSE2_Builder {
  public:
    explicit IDISA_AVX_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                               unsigned overrideNativeWidth = AVX_width)
        : IDISA_SSE2_Builder(cb, vectorWidth, laneWidth, overrideNativeWidth) {}

    std::string getBuilderCacheName() override;

    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override;
};

class IDISA_AVX2_Builder : public IDISA_AVX_Builder {
  public:
    explicit IDISA_AVX2_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                                unsigned overrideNativeWidth = AVX_width)
        : IDISA_AVX_Builder(cb, vectorWidth, laneWidth, overrideNativeWidth) {}

    std::string getBuilderCacheName() override;

    llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packss_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packh_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_add_with_carry_impl(llvm::Value *a, llvm::Value *b,
                                                                         llvm::Value *carryin) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance_impl(llvm::Value *a, llvm::Value *shiftin,
                                                                  unsigned shift) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_indexed_advance_impl(llvm::Value *a, llvm::Value *index_strm,
                                                                          llvm::Value *shiftin,
                                                                          unsigned shift) override;
    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                                    ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_srl_impl(unsigned fw, llvm::Value *a, llvm::Value *shift, const bool safe = false) override;
    llvm::Value *mvmd_sll_impl(unsigned fw, llvm::Value *a, llvm::Value *shift, const bool safe = false) override;
    llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                   ShuffleMode m = ShuffleMode::TruncateIndex) override;
    std::vector<llvm::Value *> simd_pext_impl(unsigned fw, std::vector<llvm::Value *> v,
                                              llvm::Value *extract_mask) override;
    llvm::Value *simd_pdep_impl(unsigned fw, llvm::Value *v, llvm::Value *deposit_mask) override;

    ~IDISA_AVX2_Builder() override {}
};

class IDISA_AVX512F_Builder : public IDISA_AVX2_Builder {
  public:
    explicit IDISA_AVX512F_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                                   unsigned overrideNativeWidth = AVX512_width)
        : IDISA_AVX2_Builder(cb, vectorWidth, laneWidth, overrideNativeWidth) {}

    virtual std::string getBuilderCacheName() override;

    llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packss_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_popcount_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *mvmd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) override;
    llvm::Value *mvmd_dslli_impl(unsigned fw, llvm::Value *a, llvm::Value *b, unsigned shift) override;
    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                   ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                                    ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_srl_impl(unsigned fw, llvm::Value *a, llvm::Value *shift, const bool safe) override;
    llvm::Value *mvmd_sll_impl(unsigned fw, llvm::Value *a, llvm::Value *shift, const bool safe) override;
    llvm::Value *simd_if_impl(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *simd_ternary_impl(unsigned char mask, llvm::Value *a, llvm::Value *b, llvm::Value *c) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance_impl(llvm::Value *a, llvm::Value *shiftin,
                                                                  unsigned shift) override;
};

} // namespace IDISA
