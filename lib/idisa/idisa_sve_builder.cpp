/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_sve_builder.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Host.h>

#include <arm_sve.h>

using namespace llvm;

namespace IDISA {

unsigned IDISA_SVE_Builder::NativeBitBlockWidth() {
    return codegen::HostSVEBitWidth();
}

std::string IDISA_SVE_Builder::getBuilderUniqueName() {
    return "ARM_SVE_" + std::to_string(mBitBlockWidth);
}

llvm::Value *IDISA_SVE_Builder::simd_popcount(unsigned fw, llvm::Value *a) {
    // TODO JL make a fallback for fw<8 and fw>64
    // TODO JL make a fallback for when we don't match the SVE bit width
    // ScalableVectorType::get(getInt8Ty(), fw, )
    // if ((getVectorBitWidth(a) == NativeBitBlockWidth()) && (fw >= 8) &&
    //     (fw <= 64)) {
    //     unsigned minN = ARM_SVE_min_width / fw;
    //     Function *cntIntrinsic = Intrinsic::getOrInsertDeclaration(
    //         getModule(), Intrinsic::aarch64_sve_cnt);
    //     // {ScalableVectorType::get(getInt1Ty(), minN),
    //     //  ScalableVectorType::get(getIntNTy(fw), minN)});
    //     return CreateCall(cntIntrinsic->getFunctionType(), cntIntrinsic, a);
    // } else {
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::simd_popcount(fw, a);
    });
    // }
}

llvm::Value *IDISA_SVE_Builder::simd_bitreverse(unsigned fw, llvm::Value *a) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::simd_bitreverse(fw, a);
    });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergeh(unsigned fw, llvm::Value *a,
                                             llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::esimd_mergeh(fw, a, b);
    });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergel(unsigned fw, llvm::Value *a,
                                             llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::esimd_mergel(fw, a, b);
    });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packh(unsigned fw, llvm::Value *a,
                                            llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::hsimd_packh(fw, a, b);
    });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packl(unsigned fw, llvm::Value *a,
                                            llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::hsimd_packl(fw, a, b);
    });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packus(unsigned fw, llvm::Value *a,
                                             llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::hsimd_packus(fw, a, b);
    });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle(unsigned fw,
                                             llvm::Value *data_table,
                                             llvm::Value *index_vector) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::mvmd_shuffle(fw, data_table, index_vector);
    });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle2(unsigned fw, llvm::Value *table0,
                                              llvm::Value *table1,
                                              llvm::Value *index_vector) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::mvmd_shuffle2(fw, table0, table1,
                                                index_vector);
    });
}

llvm::Value *IDISA_SVE_Builder::mvmd_compress(unsigned fw, llvm::Value *a,
                                              llvm::Value *select_mask) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::mvmd_compress(fw, a, select_mask);
    });
}

llvm::Value *IDISA_SVE_Builder::simd_sllv(unsigned fw, llvm::Value *a,
                                          llvm::Value *shifts) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::simd_sllv(fw, a, shifts);
    });
}

llvm::Value *IDISA_SVE_Builder::simd_srlv(unsigned fw, llvm::Value *a,
                                          llvm::Value *shifts) {
    // TODO JL implement
    return with_native_width(ARM_NEON_width, [=]() {
        return IDISA_ARM_Builder::simd_srlv(fw, a, shifts);
    });
}

} // namespace IDISA
