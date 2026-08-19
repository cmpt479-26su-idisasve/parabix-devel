/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_builder.h>

#include <boost/intrusive/detail/math.hpp>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <pthread.h>
#include <toolchain/toolchain.h>
#include <unistd.h>

using boost::intrusive::detail::floor_log2;

// #define PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM

using namespace llvm;

namespace IDISA {

std::string IDISA_Experiment;
static cl::opt<std::string, true> IDISA_Experiment_Option("idisa_experiment", cl::location(IDISA::IDISA_Experiment),
                                                          cl::init(""), cl::cat(codegen::CodeGenOptions));

bool isStreamTy(const Type *const t) {
    return isa<FixedVectorType>(t) && (cast<FixedVectorType>(t)->getNumElements() == 0);
}

bool isStreamSetTy(const Type *const t) { return t->isArrayTy() && (isStreamTy(t->getArrayElementType())); }

unsigned getNumOfStreams(const Type *const t) {
    if (isStreamTy(t))
        return 1;
    assert(isStreamSetTy(t));
    return cast<ArrayType>(t)->getNumElements();
}

unsigned getStreamFieldWidth(const Type *const t) {
    if (isStreamTy(t))
        return t->getScalarSizeInBits();
    assert(isStreamSetTy(t));
    return cast<ArrayType>(t)->getElementType()->getScalarSizeInBits();
}

unsigned getVectorBitWidth(Value *a) { return a->getType()->getPrimitiveSizeInBits(); }

[[noreturn]] void IDISA_Builder::UnsupportedFieldWidthError(const unsigned fw, std::string op_name) {
    report_fatal_error(StringRef(op_name) + ": Unsupported field width: " + std::to_string(fw));
}

Constant *IDISA_Builder::bit_interleave_byteshuffle_table(unsigned fw) {
    const unsigned fieldCount = getBitBlockWidth() / 8;
    if (fw > 2)
        report_fatal_error("bit_interleave_byteshuffle_table requires fw == 1 or fw == 2");
    // Bit interleave using shuffle.
    // Make a shuffle table that translates the lower 4 bits of each byte in
    // order to spread out the bits: xxxxdcba => .d.c.b.a (fw = 1)
    SmallVector<Constant *, 64> bit_interleave(fieldCount);
    for (unsigned i = 0; i < fieldCount; i++) {
        if (fw == 1)
            bit_interleave[i] = mCB->getInt8((i & 1) | ((i & 2) << 1) | ((i & 4) << 2) | ((i & 8) << 3));
        else
            bit_interleave[i] = mCB->getInt8((i & 3) | ((i & 0x0C) << 2));
    }
    return ConstantVector::get(bit_interleave);
}

Value *IDISA_Builder::fwCast(const unsigned fw, Value *const a) {
    unsigned vecWidth = getVectorBitWidth(a);
    Type *vecTy = FixedVectorType::get(mCB->getIntNTy(fw), vecWidth / fw);
    if (a->getType() == vecTy)
        return a;
    return mCB->CreateBitCast(a, vecTy);
}

Constant *IDISA_Builder::getSplat(const unsigned fieldCount, Constant *Elt) {
    return ConstantVector::getSplat(ElementCount::get(fieldCount, false), Elt);
}

Constant *IDISA_Builder::getConstantVectorSequence(unsigned fw, unsigned first, unsigned last, unsigned by) {
    const unsigned seqLgth = (last - first) / by + 1;
    assert(((first + (seqLgth - 1) * by) == last) && "invalid element sequence");
    Type *fwTy = mCB->getIntNTy(fw);
    SmallVector<Constant *, 16> elements(seqLgth);
    for (unsigned i = 0; i < seqLgth; i++) {
        elements[i] = ConstantInt::get(fwTy, i * by + first);
    }
    return ConstantVector::get(elements);
}

Constant *IDISA_Builder::getRepeatingConstantVectorSequence(unsigned fw, unsigned repeat, unsigned first, unsigned last,
                                                            unsigned by) {
    const unsigned seqLgth = (last - first) / by + 1;
    assert(((first + (seqLgth - 1) * by) == last) && "invalid element sequence");
    Type *fwTy = mCB->getIntNTy(fw);
    SmallVector<Constant *, 16> elements(seqLgth * repeat);
    for (unsigned i = 0; i < seqLgth; i++) {
        Constant *c = ConstantInt::get(fwTy, i * by + first);
        for (unsigned j = 0; j < repeat; j++) {
            elements[i + j * seqLgth] = c;
        }
    }
    return ConstantVector::get(elements);
}

Value *IDISA_Builder::CreateHalfVectorHigh(Value *vec) {
    Value *v = fwCast(mLaneWidth, vec);
    const unsigned N = getVectorBitWidth(v) / mLaneWidth;
    return mCB->CreateShuffleVector(v, UndefValue::get(v->getType()), getConstantVectorSequence(32, N / 2, N - 1));
}

Value *IDISA_Builder::CreateHalfVectorLow(Value *vec) {
    Value *v = fwCast(mLaneWidth, vec);
    const unsigned N = getVectorBitWidth(v) / mLaneWidth;
    return mCB->CreateShuffleVector(v, UndefValue::get(v->getType()), getConstantVectorSequence(32, 0, N / 2 - 1));
}

Value *IDISA_Builder::CreateDoubleVector(Value *lo, Value *hi) {
    const unsigned N = getVectorBitWidth(lo) / mLaneWidth;
    return mCB->CreateShuffleVector(fwCast(mLaneWidth, lo), fwCast(mLaneWidth, hi),
                                    getConstantVectorSequence(32, 0, 2 * N - 1));
}

Constant *IDISA_Builder::simd_himask(unsigned fw) { return simd_himask(mBitBlockWidth, fw); }

Constant *IDISA_Builder::simd_lomask(unsigned fw) { return simd_lomask(mBitBlockWidth, fw); }

Constant *IDISA_Builder::simd_himask(unsigned vector_width, unsigned fw) {
    return getSplat(vector_width / fw,
                    Constant::getIntegerValue(mCB->getIntNTy(fw), APInt::getHighBitsSet(fw, fw / 2)));
}

Constant *IDISA_Builder::simd_lomask(unsigned vector_width, unsigned fw) {
    return getSplat(vector_width / fw, Constant::getIntegerValue(mCB->getIntNTy(fw), APInt::getLowBitsSet(fw, fw / 2)));
}

Value *IDISA_Builder::simd_select_hi(unsigned fw, Value *a) {
    return simd_and(a, simd_himask(getVectorBitWidth(a), fw));
}

Value *IDISA_Builder::simd_select_lo(unsigned fw, Value *a) {
    return simd_and(a, simd_lomask(getVectorBitWidth(a), fw));
}

//
// Return a logic expression in terms of bitwise And, Or and Not for an
// arbitrary two-operand binary function corresponding to a 4-bit truth table mask.
// The 4-bit mask xyzw specifies the two-operand function fn defined by
// the following table.
//  bit_1  bit_0   fn
//    0      0     w
//    0      1     z
//    1      0     y
//    1      1     x
Value *IDISA_Builder::simd_binary(unsigned char truth_table_mask, Value *bit_1, Value *bit_0) {
    assert(bit_1->getType() == bit_0->getType());
    switch (truth_table_mask) {
    case 0x00:
        return Constant::getNullValue(bit_1->getType());
    case 0x01:
        return mCB->CreateNot(mCB->CreateOr(bit_1, bit_0));
    case 0x02:
        return mCB->CreateAnd(mCB->CreateNot(bit_1), bit_0);
    case 0x03:
        return mCB->CreateNot(bit_1);
    case 0x04:
        return mCB->CreateAnd(bit_1, mCB->CreateNot(bit_0));
    case 0x05:
        return mCB->CreateNot(bit_0);
    case 0x06:
        return mCB->CreateXor(bit_1, bit_0);
    case 0x07:
        return mCB->CreateNot(mCB->CreateAnd(bit_1, bit_0));
    case 0x08:
        return mCB->CreateAnd(bit_1, bit_0);
    case 0x09:
        return mCB->CreateNot(mCB->CreateXor(bit_1, bit_0));
    case 0x0A:
        return bit_0;
    case 0x0B:
        return mCB->CreateOr(mCB->CreateNot(bit_1), bit_0);
    case 0x0C:
        return bit_1;
    case 0x0D:
        return mCB->CreateOr(bit_1, mCB->CreateNot(bit_0));
    case 0x0E:
        return mCB->CreateOr(bit_1, bit_0);
    case 0x0F:
        return Constant::getAllOnesValue(bit_1->getType());
    default:
        report_fatal_error("simd_binary mask is in wrong format!");
    }
}

Value *IDISA_Builder::bitblock_popcount(Value *const to_count) {
    const auto fieldWidth = mCB->getSizeTy()->getBitWidth();
    auto fields = (getBitBlockWidth() / fieldWidth);
    Value *fieldCounts = simd_popcount(fieldWidth, to_count);
    while (fields > 1) {
        fields /= 2;
        fieldCounts = mCB->CreateAdd(fieldCounts, mvmd_srli(fieldWidth, fieldCounts, fields));
    }
    return mvmd_extract(fieldWidth, fieldCounts, 0);
}

Value *IDISA_Builder::simd_and(Value *a, Value *b, StringRef s) {
    return a->getType() == b->getType() ? mCB->CreateAnd(a, b, s) : mCB->CreateAnd(bitCast(a), bitCast(b), s);
}

Value *IDISA_Builder::simd_or(Value *a, Value *b, StringRef s) {
    return a->getType() == b->getType() ? mCB->CreateOr(a, b, s) : mCB->CreateOr(bitCast(a), bitCast(b), s);
}

Value *IDISA_Builder::simd_xor(Value *a, Value *b, StringRef s) {
    return a->getType() == b->getType() ? mCB->CreateXor(a, b, s) : mCB->CreateXor(bitCast(a), bitCast(b), s);
}

Value *IDISA_Builder::simd_not(Value *a, StringRef s) {
    return simd_xor(a, Constant::getAllOnesValue(a->getType()), s);
}

} // namespace IDISA
