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

unsigned IDISA_SVE_Builder::NativeBitBlockWidth() {
#if 0
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
        unsigned svN = mNativeBitBlockWidth / fw;
        IntegerType *fTy = getIntNTy(fw);
        FixedVectorType *vTy = FixedVectorType::get(fTy, svN);
        ScalableVectorType *svTy = ScalableVectorType::get(fTy, svN);
        ScalableVectorType *svPTy = ScalableVectorType::get(getInt1Ty(), svN);

        Function *svePtrue = Intrinsic::getOrInsertDeclaration(getModule(), Intrinsic::aarch64_sve_ptrue, {svPTy});
        // Function *svePtrue = Intrinsic::getOrInsertDeclaration(
        //     getModule(), Intrinsic::aarch64_sve_whilelo, {vTy, svTy});

        Function *vectorExtract =
            Intrinsic::getOrInsertDeclaration(getModule(), Intrinsic::vector_extract, {vTy, svTy});
        Function *vectorInsert = Intrinsic::getOrInsertDeclaration(getModule(), Intrinsic::vector_insert, {svTy, vTy});
        Function *sveCnt = Intrinsic::getOrInsertDeclaration(getModule(), Intrinsic::aarch64_sve_cnt, {svTy});
        // Constant *i64Zero = ConstantInt::get(getIntNTy(64), 0);
        Constant *i64Zero = Constant::getNullValue(getIntNTy(64));
        svpattern pattern;
        // clang-format off
        switch (svN) {
            case 1:   pattern = SV_VL1;   break;
            case 2:   pattern = SV_VL2;   break;
            case 4:   pattern = SV_VL4;   break;
            case 8:   pattern = SV_VL8;   break;
            case 16:  pattern = SV_VL16;  break;
            case 32:  pattern = SV_VL32;  break;
            case 64:  pattern = SV_VL64;  break;
            case 128: pattern = SV_VL128; break;
            case 256: pattern = SV_VL256; break;
            default:  pattern = SV_ALL;   break;
        }
        // clang-format on
        if (pattern == SV_ALL) {
            report_fatal_error(StringRef("simd_popcount: Vector size has no predicate pattern: ") +
                               std::to_string(svN));
        }
        Constant *i32PredPat = ConstantInt::get(getIntNTy(32), pattern);

        Value *v1 = CreateCall(vectorInsert->getFunctionType(), vectorInsert, {UndefValue::get(svTy), a, i64Zero});
        Value *pred = CreateCall(svePtrue->getFunctionType(), svePtrue, {i32PredPat});
        Value *v2 = CreateCall(sveCnt->getFunctionType(), sveCnt, {UndefValue::get(svTy), pred, v1});
        Value *v3 = CreateCall(vectorExtract->getFunctionType(), vectorExtract, {v2, i64Zero});

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
