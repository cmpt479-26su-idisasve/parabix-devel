#pragma once

#include <idisa/idisa_arm_builder.h>
#include <idisa/idisa_builder.h>

namespace IDISA {

constexpr unsigned ARM_SVE_min_width = 128;

class IDISA_SVE_Builder : public IDISA_ARM_Builder {
  public:
    unsigned NativeBitBlockWidth();

    virtual std::string getBuilderUniqueName() override;

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
    template <class F>
    llvm::Value *with_native_width(unsigned temp_width, F &&f) {
        unsigned real_width = mNativeBitBlockWidth;
        try {
            const_cast<unsigned &>(mNativeBitBlockWidth) = temp_width;
            llvm::Value *result = f();
            const_cast<unsigned &>(mNativeBitBlockWidth) = real_width;
            return result;
        } catch (...) {
            const_cast<unsigned &>(mNativeBitBlockWidth) = real_width;
            throw;
        }
    }
};

} // namespace IDISA
