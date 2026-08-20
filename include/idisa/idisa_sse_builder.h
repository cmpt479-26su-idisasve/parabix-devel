#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_builder.h>

namespace IDISA {

constexpr unsigned SSE_width = 128;
    
class IDISA_SSE_Builder : public virtual IDISA_Builder {
public:
    static constexpr unsigned NativeBitBlockWidth() {return SSE_width;}

    IDISA_SSE_Builder(): IDISA_Builder(DONTUSE_CONSTRUCTOR()) {}

    virtual std::string getBuilderCacheName() override;
    llvm::Value * hsimd_signmask(unsigned fw, llvm::Value * a) override;
    llvm::Value * mvmd_compress(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
};

class IDISA_SSE2_Builder : public IDISA_SSE_Builder {
public:
    using IDISA_SSE_Builder::NativeBitBlockWidth;

    IDISA_SSE2_Builder(): IDISA_Builder(DONTUSE_CONSTRUCTOR()) {}

    virtual std::string getBuilderCacheName() override;
    llvm::Value * hsimd_signmask(unsigned fw, llvm::Value * a) override;
    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packus(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex) override;
    virtual std::vector<llvm::Value *> simd_pext(unsigned fw, std::vector<llvm::Value *>, llvm::Value * extract_mask) override;
};

class IDISA_SSSE3_Builder : public IDISA_SSE2_Builder {
public:
    using IDISA_SSE_Builder::NativeBitBlockWidth;

    IDISA_SSSE3_Builder(): IDISA_Builder(DONTUSE_CONSTRUCTOR()) {}

    virtual std::string getBuilderCacheName() override;
    llvm::Value * esimd_mergeh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * esimd_mergel(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex) override;
    llvm::Value * mvmd_compress(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
    llvm::Value * mvmd_expand(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
};

}

