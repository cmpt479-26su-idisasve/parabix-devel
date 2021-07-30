#ifndef IDISA_ARM_BUILDER_H
#define IDISA_ARM_BUILDER_H

#include <idisa/idisa_builder.h>
#if defined(__ARM_ARCH)
#include<arm_neon.h>
#endif

namespace IDISA {

const unsigned ARM_width = 32;
    
class IDISA_ARM_Builder : public virtual IDISA_Builder {
public:
    static const unsigned NativeBitBlockWidth = ARM_width;
    IDISA_ARM_Builder(llvm::LLVMContext & C, unsigned bitBlockWidth, unsigned laneWidth)
    : IDISA_Builder(C, ARM_width, bitBlockWidth, laneWidth) {

    }

    virtual std::string getBuilderUniqueName() override;
    #if defined(__ARM_ARCH)
    int arm_signmak();
    llvm::Value * hsimd_signmask(unsigned fw, llvm::Value * a) override;
    #endif

    // SSE
    /*
    llvm::Value * mvmd_compress(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
    // SSE2
    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packus(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance(llvm::Value * a, llvm::Value * shiftin, unsigned shift) override;
    llvm::Value * mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) override;
    // SSSE3
    llvm::Value * esimd_mergeh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * esimd_mergel(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    */

    ~IDISA_ARM_Builder() {}
};

}

#endif // IDISA_ARM_BUILDER_H
