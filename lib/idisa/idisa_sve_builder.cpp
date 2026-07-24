/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_sve_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Host.h>

#include <arm_sve.h>

using namespace llvm;

namespace IDISA {

__attribute__((target ("+sve")))
unsigned HostSVEBitWidth() {
    return svcntb() * 8;
}

IDISA_SVE_Builder::IDISA_SVE_Builder(llvm::LLVMContext& C,
                                     const codegen::FeatureSet& featureSet,
                                     unsigned bitBlockWidth, unsigned laneWidth)
    : IDISA_Builder(C, featureSet, HostSVEBitWidth(), bitBlockWidth, laneWidth),
      IDISA_ARM_Builder(C, featureSet, bitBlockWidth, laneWidth) {}

std::string IDISA_SVE_Builder::getBuilderUniqueName() {
    return "ARM_SVE_" + std::to_string(mBitBlockWidth);
}

llvm::Value * IDISA_SVE_Builder::simd_popcount(unsigned fw, llvm::Value* a) {
    // TODO JL implement
    return IDISA_ARM_Builder::simd_popcount(fw, a);
}

llvm::Value * IDISA_SVE_Builder::simd_bitreverse(unsigned fw, llvm::Value* a) {
    // TODO JL implement
    return IDISA_ARM_Builder::simd_bitreverse(fw, a);
}

llvm::Value * IDISA_SVE_Builder::esimd_mergeh(unsigned fw, llvm::Value * a, llvm::Value * b) {
    // TODO JL implement
    return IDISA_ARM_Builder::esimd_mergeh(fw, a, b);
}

llvm::Value * IDISA_SVE_Builder::esimd_mergel(unsigned fw, llvm::Value * a, llvm::Value * b) {
    // TODO JL implement
    return IDISA_ARM_Builder::esimd_mergel(fw, a, b);
}

llvm::Value * IDISA_SVE_Builder::hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) {
    // TODO JL implement
    return IDISA_ARM_Builder::hsimd_packh(fw, a, b);
}

llvm::Value * IDISA_SVE_Builder::hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) {
    // TODO JL implement
    return IDISA_ARM_Builder::hsimd_packl(fw, a, b);
}

llvm::Value * IDISA_SVE_Builder::hsimd_packus(unsigned fw, llvm::Value * a, llvm::Value * b) {
    // TODO JL implement
    return IDISA_ARM_Builder::hsimd_packus(fw, a, b);
}

llvm::Value * IDISA_SVE_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {
    // TODO JL implement
    return IDISA_ARM_Builder::mvmd_shuffle(fw, data_table, index_vector);
}

llvm::Value * IDISA_SVE_Builder::mvmd_shuffle2(unsigned fw, llvm::Value * table0, llvm::Value * table1, llvm::Value * index_vector) {
    // TODO JL implement
    return IDISA_ARM_Builder::mvmd_shuffle2(fw, table0, table1, index_vector);
}

}
