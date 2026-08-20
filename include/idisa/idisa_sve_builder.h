#pragma once

#include <idisa/idisa_arm_builder.h>
#include <idisa/idisa_builder.h>

namespace IDISA {

constexpr unsigned ARM_SVE_min_width = 128;

class IDISA_SVE_Builder : public IDISA_ARM_Builder {
  public:
    unsigned NativeBitBlockWidth();

    virtual std::string getBuilderCacheName() override;

    llvm::Value *simd_popcount(unsigned fw, llvm::Value *a) override;
    llvm::Value *simd_bitreverse(unsigned fw, llvm::Value *a) override;
    llvm::Value *esimd_mergeh(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *esimd_mergel(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packh(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packl(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *hsimd_packus(unsigned fw, llvm::Value *a, llvm::Value *b) override;
    llvm::Value *mvmd_shuffle(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                              ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_shuffle2(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value *mvmd_compress(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;
    llvm::Value *mvmd_expand(unsigned fw, llvm::Value *a, llvm::Value *select_mask) override;

    IDISA_SVE_Builder() : IDISA_Builder(DONTUSE_CONSTRUCTOR()) {}
    ~IDISA_SVE_Builder() = default;

  private:
    template <class F> llvm::Value *withNativeWidth(unsigned tempWidth, F &&f);
    template <class F> llvm::Value *encapsulateScalableUnary(unsigned fw, llvm::Value *param, F &&createOp);
    template <class F>
    llvm::Value *encapsulateScalableBinary(unsigned fw, llvm::Value *param1, llvm::Value *param2, F &&createOp);
};

} // namespace IDISA
