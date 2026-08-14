#pragma once

#include <idisa/idisa_builder.h>

namespace IDISA {

constexpr unsigned ARM_Neon_width = 128;

class IDISA_ARM_Builder : public virtual IDISA_Builder {
public:
    static constexpr unsigned NativeBitBlockWidth() {return ARM_Neon_width;}

    IDISA_ARM_Builder(): IDISA_Builder(DONTUSE_CONSTRUCTOR()) {}
    ~IDISA_ARM_Builder() = default;

    virtual std::string getBuilderUniqueName() override;
    llvm::Value* simd_popcount(unsigned fw, llvm::Value* a) override;
    llvm::Value* simd_bitreverse(unsigned fw, llvm::Value* a) override;
    llvm::Value * esimd_mergeh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * esimd_mergel(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packus(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value * mvmd_shuffle2(unsigned fw, llvm::Value * table0, llvm::Value * table1, llvm::Value * index_vector,
                                ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value * mvmd_compress(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
    llvm::Value * mvmd_expand(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
    
protected:
    llvm::Value * byteMaskToLaneMask(llvm::Value * byteMask);
    llvm::Value * compressBytes(llvm::Value * a, llvm::Value * byteMask);
    llvm::Value * fieldPermute(unsigned fw, llvm::Value * a, llvm::Value * select_mask, bool isExpand);
};

}
