#pragma once

#include <idisa/idisa_generic_builder.h>

namespace IDISA {

constexpr unsigned Neon_width = 128;

class IDISA_Neon_Builder : public IDISA_Generic_Builder {
  public:
    explicit IDISA_Neon_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth)
        : IDISA_Generic_Builder(cb, vectorWidth, laneWidth, Neon_width, 64, 8) {}
    ~IDISA_Neon_Builder() = default;

    std::string getBuilderCacheName() override;

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

  protected:
    llvm::Value *compressBytes(llvm::Value *a, llvm::Value *byteMask);
    llvm::Value *fieldPermute(unsigned fw, llvm::Value *a, llvm::Value *select_mask, bool isExpand);
};

} // namespace IDISA
