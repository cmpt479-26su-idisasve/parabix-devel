#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_generic_builder.h>

namespace IDISA {

constexpr unsigned SSE_width = 128;

class IDISA_SSE_Builder : public IDISA_Generic_Builder {
  public:
    explicit IDISA_SSE_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                               unsigned overrideNativeWidth = SSE_width)
        : IDISA_Generic_Builder(cb, vectorWidth, laneWidth, overrideNativeWidth) {}

    std::string getBuilderCacheName() override;

    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
};

class IDISA_SSE2_Builder : public IDISA_SSE_Builder {
  public:
    explicit IDISA_SSE2_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                                unsigned overrideNativeWidth = SSE_width)
        : IDISA_SSE_Builder(cb, vectorWidth, laneWidth, overrideNativeWidth) {}

    std::string getBuilderCacheName() override;

    llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) override;
    llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                   ShuffleMode m = ShuffleMode::TruncateIndex) override;
    virtual std::vector<llvm::Value *> simd_pext_impl(unsigned fw, std::vector<llvm::Value *>,
                                                      llvm::Value *extract_mask) override;
};

class IDISA_SSSE3_Builder : public IDISA_SSE2_Builder {
  public:
    explicit IDISA_SSSE3_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth,
                                 unsigned overrideNativeWidth = SSE_width)
        : IDISA_SSE2_Builder(cb, vectorWidth, laneWidth, overrideNativeWidth) {}

    std::string getBuilderCacheName() override;

    llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                   ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
};

} // namespace IDISA
