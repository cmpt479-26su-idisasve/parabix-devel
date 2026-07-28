#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <idisa/idisa_builder.h>

namespace IDISA {

constexpr unsigned I64_width = 64;

class IDISA_I64_Builder : public virtual IDISA_Builder {
public:
    static constexpr unsigned NativeBitBlockWidth() {return I64_width;}

    IDISA_I64_Builder(): IDISA_Builder(DONTUSE_CONSTRUCTOR()) {} 

    virtual std::string getBuilderUniqueName() override;

    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;

};

}

