#pragma once

#include <idisa/idisa_arm_builder.h>

namespace IDISA {

constexpr unsigned SVE_width = 128;

class IDISA_SVE_Builder : public IDISA_ARM_Builder {
public:
    static constexpr unsigned NativeBitBlockWidth = SVE_width;

    IDISA_SVE_Builder(llvm::LLVMContext& C, const FeatureSet& featureSet,
                      unsigned bitBlockWidth, unsigned laneWidth);

    virtual std::string getBuilderUniqueName() override;

    ~IDISA_SVE_Builder() {}
};

}

