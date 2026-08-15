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

using namespace llvm;

namespace IDISA {

static svpattern PatternForVectorLength(unsigned svN) {
    switch (svN) {
    case 1:
        return SV_VL1;
    case 2:
        return SV_VL2;
    case 4:
        return SV_VL4;
    case 8:
        return SV_VL8;
    case 16:
        return SV_VL16;
    case 32:
        return SV_VL32;
    case 64:
        return SV_VL64;
    case 128:
        return SV_VL128;
    case 256:
        return SV_VL256;
    default:
        report_fatal_error(StringRef("simd_popcount: Vector size has no predicate pattern: ") + std::to_string(svN));
    }
}

unsigned IDISA_SVE_Builder::NativeBitBlockWidth() {
#if 1
    // This unfortunately doesn't work: LLVM doesn't know what to do with big fixed vectors
    return codegen::HostSVEBitWidth();
#else
    return IDISA::ARM_Neon_width;
#endif
}

std::string IDISA_SVE_Builder::getBuilderUniqueName() { return "SVE"; }

llvm::Value *IDISA_SVE_Builder::simd_popcount(unsigned fw, llvm::Value *a) {
    // TODO JL make a fallback for fw<8 and fw>64..?
    // ScalableVectorType::get(getInt8Ty(), fw, )

    // With SVE it's actually okay to synthesize smaller chunks than "native"
    assert(mNativeBitBlockWidth <= NativeBitBlockWidth());
    assert(mNativeBitBlockWidth >= fw);

    unsigned vectorWidth = getVectorBitWidth(a);
    if (vectorWidth > mNativeBitBlockWidth) {
        // Let IDISA_Builder split the instruction up -- it should call
        // back into us with operations on smaller chunks
        return IDISA_Builder::simd_popcount(fw, a);
    }
    if ((vectorWidth <= mNativeBitBlockWidth) && (fw >= 8) && (fw <= 64)) {
        unsigned fvN = mNativeBitBlockWidth / fw;
        unsigned svN = ARM_SVE_min_width / fw;
        IntegerType *fTy = getIntNTy(fw);
        FixedVectorType *fvTy = FixedVectorType::get(fTy, fvN);
        ScalableVectorType *svTy = ScalableVectorType::get(fTy, svN);
        ScalableVectorType *svPTy = ScalableVectorType::get(getInt1Ty(), svN);

        Constant *i64Zero = Constant::getNullValue(getIntNTy(64));
        Constant *i32PredPat = ConstantInt::get(getIntNTy(32), PatternForVectorLength(fvN));

        Value *pred = CreateIntrinsic(Intrinsic::aarch64_sve_ptrue, {svPTy}, {i32PredPat});

        // Value *PoisonValue::get(svTy);
        // for (unsigned i = 0; i <) {
        //     CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvTy}, {, a, i64Zero});
        // }

        // Value *v1 = CreateIntrinsic(Intrinsic::vector_insert, {svTy, fvTy}, {PoisonValue::get(svTy), a, i64Zero});
        Value *tempP = CreateAlloca(fvTy);
        CreateStore(a, tempP);
        Value *v1 = CreateIntrinsic(Intrinsic::aarch64_sve_ld1, {svTy}, {pred, tempP});

        Value *v2 = CreateIntrinsic(Intrinsic::aarch64_sve_cnt, {svTy}, {PoisonValue::get(svTy), pred, v1});

        // Value *v3 = CreateIntrinsic(Intrinsic::vector_extract, {fvTy, svTy}, {v2, i64Zero});
        CreateIntrinsic(Intrinsic::aarch64_sve_st1, {svTy}, {v2, pred, tempP});
        Value *v3 = CreateLoad(fvTy, tempP);

        return v3;
    } else {
        return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::simd_popcount(fw, a); });
    }
}

llvm::Value *IDISA_SVE_Builder::simd_bitreverse(unsigned fw, llvm::Value *a) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::simd_bitreverse(fw, a); });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergeh(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::esimd_mergeh(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::esimd_mergel(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::esimd_mergel(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packh(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::hsimd_packh(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packl(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::hsimd_packl(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::hsimd_packus(unsigned fw, llvm::Value *a, llvm::Value *b) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::hsimd_packus(fw, a, b); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                             ShuffleMode m) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width,
                             [=]() { return IDISA_ARM_Builder::mvmd_shuffle(fw, data_table, index_vector); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_shuffle2(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                              llvm::Value *index_vector, ShuffleMode m) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width,
                             [=]() { return IDISA_ARM_Builder::mvmd_shuffle2(fw, table0, table1, index_vector); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_compress(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::mvmd_compress(fw, a, select_mask); });
}

llvm::Value *IDISA_SVE_Builder::mvmd_expand(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
    // TODO JL implement
    return with_native_width(ARM_Neon_width, [=]() { return IDISA_ARM_Builder::mvmd_expand(fw, a, select_mask); });
}

} // namespace IDISA
