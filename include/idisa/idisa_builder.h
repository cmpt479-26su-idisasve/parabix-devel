#pragma once

/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <idisa/CBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <string>
#include <toolchain/toolchain.h>

namespace llvm {
class ArrayType;
class Constant;
class FixedVectorType;
class LLVMContext;
class StoreInst;
class Value;
class VectorType;
} // namespace llvm

namespace IDISA {

extern std::string IDISA_Experiment;

bool isStreamTy(const llvm::Type *const t);

bool isStreamSetTy(const llvm::Type *const t);

unsigned getNumOfStreams(const llvm::Type *const t);

unsigned getStreamFieldWidth(const llvm::Type *const t);

unsigned getVectorBitWidth(llvm::Value *vec);

// clang-format off
#define CACHE_NAME_BUILD_ID                                               \
    {                                                                     \
        /*Year*/    __DATE__[7], __DATE__[8], __DATE__[9], __DATE__[10],  \
        /*Month*/   __DATE__[0], __DATE__[1], __DATE__[2],                \
        /*Day*/     __DATE__[4] == ' ' ? '0' : __DATE__[4], __DATE__[5],  \
                    'T',                                                  \
        /*Hour*/    __TIME__[0], __TIME__[1],                             \
        /*Min*/     __TIME__[3], __TIME__[4],                             \
        /*Sec*/     __TIME__[6], __TIME__[7],                             \
                    0                                                     \
    }

// Put this in your IDISA_Builder descendant's .cpp file to define the cache name
// e.g.: DEFINE_BUILDER_CACHE_NAME(IDISA_ARM_Builder, "ARM_Neon", ARM_Neon_width)
// The override for makeCacheName must still be declared in the header.
#define DEFINE_BUILDER_CACHE_NAME(classname, basename, nativebits)       \
    ::std::string classname::getBuilderCacheName() {                     \
        static constexpr char const cacheName[] = CACHE_NAME_BUILD_ID;   \
        std::string name = basename;                                     \
        if(getBitBlockWidth() != nativebits) {                           \
            name += '_';                                                 \
            name += ::std::to_string(getBitBlockWidth() );               \
        }                                                                \
        name += '_';                                                     \
        name += cacheName;                                               \
        return name;                                                     \
    }
// clang-format on

// The following shuffle modes control what happens when an index value exceeds
// the number of entries in the given table.
enum class ShuffleMode { TruncateIndex, ZeroOnIndexOver, ZeroOnHighIndexBit };

class IDISA_Builder {
  public:
    virtual ~IDISA_Builder() = default;

    unsigned getBitBlockWidth() const { return mBitBlockWidth; }
    unsigned getLaneWidth() const { return mLaneWidth; }
    llvm::IntegerType *getLaneTy() const { return mCB->getIntNTy(mLaneWidth); }
    llvm::FixedVectorType *getBitBlockType() const { return mBitBlockType; }
    llvm::Constant *allZeroes() const { return mZeroInitializer; }
    llvm::Constant *allOnes() const { return mOneInitializer; }

    CBuilder &getCBuilder() { return *mCB; }
    llvm::LLVMContext &getContext() { return mCB->getContext(); }

    llvm::Value *bitCast(llvm::Value *a) { return fwCast(mLaneWidth, a); }

    llvm::Constant *getConstantVectorSequence(unsigned fw, unsigned first, unsigned last, unsigned by = 1);

    llvm::Constant *getRepeatingConstantVectorSequence(unsigned fw, unsigned repeat, unsigned first, unsigned last,
                                                       unsigned by = 1);

    llvm::Value *CreateHalfVectorHigh(llvm::Value *);

    llvm::Value *CreateHalfVectorLow(llvm::Value *);

    llvm::Value *CreateDoubleVector(llvm::Value *lo, llvm::Value *hi);

    llvm::Constant *getSplat(const unsigned fieldCount, llvm::Constant *Elt);
    llvm::Constant *getSplat(const unsigned fieldCount, llvm::APInt intVal);
    llvm::Constant *getSplatN(const unsigned fw, const unsigned fieldCount, int intVal);

    llvm::LoadInst *CreateBlockAlignedLoad(llvm::Type *type, llvm::Value *const ptr) {
        return mCB->CreateAlignedLoad(type, ptr, mBitBlockWidth / 8);
    }

    llvm::StoreInst *CreateBlockAlignedStore(llvm::Value *const value, llvm::Value *const ptr) {
        return mCB->CreateAlignedStore(value, ptr, mBitBlockWidth / 8);
    }

    llvm::Value *CreateBlockAlignedMalloc(llvm::Value *size) {
        return mCB->CreateAlignedMalloc(size, mBitBlockWidth / 8);
    }

    llvm::VectorType *fwVectorType(const unsigned fw) {
        return llvm::FixedVectorType::get(mCB->getIntNTy(fw), mBitBlockWidth / fw);
    }

    llvm::VectorType *singletonVectorType(const unsigned fw) {
        return llvm::FixedVectorType::get(mCB->getIntNTy(fw), 1);
    }

    static llvm::FixedVectorType *LLVM_READNONE getStreamTy(llvm::LLVMContext &C, const unsigned FieldWidth = 1) {
        return llvm::FixedVectorType::get(llvm::IntegerType::getIntNTy(C, FieldWidth), static_cast<unsigned>(0));
    }

    static llvm::ArrayType *LLVM_READNONE getStreamSetTy(llvm::LLVMContext &C, const unsigned NumElements = 1,
                                                         const unsigned FieldWidth = 1) {
        return llvm::ArrayType::get(getStreamTy(C, FieldWidth), NumElements);
    }

    llvm::FixedVectorType *getStreamTy(const unsigned FieldWidth = 1) { return getStreamTy(getContext(), FieldWidth); }

    llvm::ArrayType *getStreamSetTy(const unsigned NumElements = 1, const unsigned FieldWidth = 1) {
        return getStreamSetTy(getContext(), NumElements, FieldWidth);
    }

    llvm::Constant *simd_himask(unsigned fw);
    llvm::Constant *simd_lomask(unsigned fw);

    llvm::Constant *simd_himask(unsigned vector_width, unsigned fw);
    llvm::Constant *simd_lomask(unsigned vector_width, unsigned fw);

    llvm::Value *simd_select_hi(unsigned fw, llvm::Value *a);
    llvm::Value *simd_select_lo(unsigned fw, llvm::Value *a);

    llvm::Value *simd_fill(unsigned fw, llvm::Value *a) { return simd_fill_impl(fw, a); }

    llvm::Value *simd_fill(unsigned vector_width, unsigned fw, llvm::Value *a) {
        return simd_fill_impl(vector_width, fw, a);
    }

    llvm::Value *simd_add(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_add_impl(fw, a, b); }
    llvm::Value *simd_sub(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_sub_impl(fw, a, b); }
    llvm::Value *simd_mult(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_mult_impl(fw, a, b); }
    llvm::Value *simd_eq(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_eq_impl(fw, a, b); }
    llvm::Value *simd_ne(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_ne_impl(fw, a, b); }
    llvm::Value *simd_gt(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_gt_impl(fw, a, b); }
    llvm::Value *simd_ge(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_ge_impl(fw, a, b); }
    llvm::Value *simd_lt(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_lt_impl(fw, a, b); }
    llvm::Value *simd_le(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_le_impl(fw, a, b); }
    llvm::Value *simd_ugt(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_ugt_impl(fw, a, b); }
    llvm::Value *simd_ult(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_ult_impl(fw, a, b); }
    llvm::Value *simd_ule(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_ule_impl(fw, a, b); }
    llvm::Value *simd_uge(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_uge_impl(fw, a, b); }
    llvm::Value *simd_max(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_max_impl(fw, a, b); }
    llvm::Value *simd_umax(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_umax_impl(fw, a, b); }
    llvm::Value *simd_min(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_min_impl(fw, a, b); }
    llvm::Value *simd_umin(unsigned fw, llvm::Value *a, llvm::Value *b) { return simd_umin_impl(fw, a, b); }
    llvm::Value *simd_if(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b) {
        return simd_if_impl(fw, cond, a, b);
    }

    // Return a logic expression in terms of bitwise And, Or and Not for an
    // arbitrary two-operand boolean function corresponding to a 4-bit truth table mask.
    // The 4-bit mask dcba specifies the two-operand function fn defined by
    // the following table.
    //  bit_1  bit_0   fn
    //    0      0     a
    //    0      1     b
    //    1      0     c
    //    1      1     d
    llvm::Value *simd_binary(unsigned char mask, llvm::Value *bit_1, llvm::Value *bit_0);

    // Return a logic expression in terms of bitwise And, Or and Not for an
    // arbitrary three-operand boolean function corresponding to an 8-bit truth table mask.
    // The 8-bit mask hgfedcba specifies the three-operand function fn defined by
    // the following table.
    //  bit_2  bit_1  bit_0   fn
    //    0      0      0     a
    //    0      0      1     b
    //    0      1      0     c
    //    0      1      1     d
    //    1      0      0     e
    //    1      0      1     f
    //    1      1      0     g
    //    1      1      1     h
    llvm::Value *simd_ternary(unsigned char mask, llvm::Value *bit_2, llvm::Value *bit_1, llvm::Value *bit_0) {
        return simd_ternary_impl(mask, bit_2, bit_1, bit_0);
    }

    llvm::Value *simd_slli(unsigned fw, llvm::Value *a, unsigned shift) { return simd_slli_impl(fw, a, shift); }
    llvm::Value *simd_srli(unsigned fw, llvm::Value *a, unsigned shift) { return simd_srli_impl(fw, a, shift); }
    llvm::Value *simd_srai(unsigned fw, llvm::Value *a, unsigned shift) { return simd_srai_impl(fw, a, shift); }
    llvm::Value *simd_sllv(unsigned fw, llvm::Value *a, llvm::Value *shifts) { return simd_sllv_impl(fw, a, shifts); }
    llvm::Value *simd_srlv(unsigned fw, llvm::Value *a, llvm::Value *shifts) { return simd_srlv_impl(fw, a, shifts); }
    llvm::Value *simd_rotl(unsigned fw, llvm::Value *a, llvm::Value *rotates) { return simd_rotl_impl(fw, a, rotates); }
    llvm::Value *simd_rotr(unsigned fw, llvm::Value *a, llvm::Value *rotates) { return simd_rotr_impl(fw, a, rotates); }

    std::vector<llvm::Value *> simd_pext(unsigned fw, std::vector<llvm::Value *> vs, llvm::Value *extract_mask) {
        return simd_pext_impl(fw, vs, extract_mask);
    }
    llvm::Value *simd_pext(unsigned fw, llvm::Value *v, llvm::Value *extract_mask) {
        return simd_pext(fw, std::vector<llvm::Value *>{v}, extract_mask)[0];
    }
    llvm::Value *simd_pdep(unsigned fw, llvm::Value *v, llvm::Value *deposit_mask) {
        return simd_pdep_impl(fw, v, deposit_mask);
    }
    llvm::Value *simd_any(unsigned fw, llvm::Value *a) { return simd_any_impl(fw, a); }
    llvm::Value *simd_popcount(unsigned fw, llvm::Value *a) { return simd_popcount_impl(fw, a); }
    llvm::Value *hsimd_partial_sum(unsigned fw, llvm::Value *a) { return hsimd_partial_sum_impl(fw, a); }
    llvm::Value *simd_cttz(unsigned fw, llvm::Value *a) { return simd_cttz_impl(fw, a); }

    llvm::Value *simd_bitreverse(unsigned fw, llvm::Value *a) { return simd_bitreverse_impl(fw, a); }

    llvm::Value *esimd_mergeh(unsigned fw, llvm::Value *a, llvm::Value *b) { return esimd_mergeh_impl(fw, a, b); }
    llvm::Value *esimd_mergel(unsigned fw, llvm::Value *a, llvm::Value *b) { return esimd_mergel_impl(fw, a, b); }
    llvm::Value *esimd_bitspread(unsigned vec_width, unsigned fw, llvm::Value *bitmask) {
        return esimd_bitspread_impl(vec_width, fw, bitmask);
    }

    llvm::Value *hsimd_packh(unsigned fw, llvm::Value *a, llvm::Value *b) { return hsimd_packh_impl(fw, a, b); }
    llvm::Value *hsimd_packl(unsigned fw, llvm::Value *a, llvm::Value *b) { return hsimd_packl_impl(fw, a, b); }
    // Pack signed values with signed/unsigned saturation.
    llvm::Value *hsimd_packss(unsigned fw, llvm::Value *a, llvm::Value *b) { return hsimd_packss_impl(fw, a, b); }
    llvm::Value *hsimd_packus(unsigned fw, llvm::Value *a, llvm::Value *b) { return hsimd_packus_impl(fw, a, b); }
    llvm::Value *hsimd_packh_in_lanes(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) {
        return hsimd_packh_in_lanes_impl(lanes, fw, a, b);
    }
    llvm::Value *hsimd_packl_in_lanes(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) {
        return hsimd_packl_in_lanes_impl(lanes, fw, a, b);
    }

    llvm::Value *hsimd_signmask(unsigned fw, llvm::Value *a) {
        // For very large bitBlockWidth (which would happen naturally on RVV with 2048-bit SIMD registers),
        // hsimd_signmask can in theory produce ints that are larger than sizeTy
        assert(getBitBlockWidth() / fw <= mCB->getSizeTy()->getBitWidth());
        return hsimd_signmask_impl(fw, a);
    }

    llvm::Value *mvmd_extract(unsigned fw, llvm::Value *a, unsigned fieldIndex) {
        return mvmd_extract_impl(fw, a, fieldIndex);
    }
    llvm::Value *mvmd_insert(unsigned fw, llvm::Value *blk, llvm::Value *elt, unsigned fieldIndex) {
        return mvmd_insert_impl(fw, blk, elt, fieldIndex);
    }

    llvm::Value *mvmd_sll(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe = false) {
        return mvmd_sll_impl(fw, value, shift, safe);
    }
    llvm::Value *mvmd_srl(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe = false) {
        return mvmd_srl_impl(fw, value, shift, safe);
    }
    llvm::Value *mvmd_slli(unsigned fw, llvm::Value *a, unsigned shift) { return mvmd_slli_impl(fw, a, shift); }
    llvm::Value *mvmd_srli(unsigned fw, llvm::Value *a, unsigned shift) { return mvmd_srli_impl(fw, a, shift); }
    llvm::Value *mvmd_dslli(unsigned fw, llvm::Value *a, llvm::Value *b, unsigned shift) {
        return mvmd_dslli_impl(fw, a, b, shift);
    }
    llvm::Value *mvmd_dsll(unsigned fw, llvm::Value *a, llvm::Value *b, llvm::Value *shift) {
        return mvmd_dsll_impl(fw, a, b, shift);
    }

    // The following shuffle modes control what happens when an index value exceeds
    // the number of entries in the given table.
    llvm::Value *mvmd_shuffle(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                              ShuffleMode m = ShuffleMode::TruncateIndex) {
        return mvmd_shuffle_impl(fw, data_table, index_vector, m);
    }
    llvm::Value *mvmd_shuffle2(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex) {
        return mvmd_shuffle2_impl(fw, table0, table1, index_vector, m);
    }
    llvm::Value *mvmd_compress(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
        return mvmd_compress_impl(fw, a, select_mask);
    }
    llvm::Value *mvmd_expand(unsigned fw, llvm::Value *a, llvm::Value *select_mask) {
        return mvmd_expand_impl(fw, a, select_mask);
    }

    llvm::Value *bitblock_any(llvm::Value *a) { return bitblock_any_impl(a); }
    // full add producing {carryout, sum}
    std::pair<llvm::Value *, llvm::Value *> bitblock_add_with_carry(llvm::Value *a, llvm::Value *b,
                                                                    llvm::Value *carryin) {
        return bitblock_add_with_carry_impl(a, b, carryin);
    }
    std::pair<llvm::Value *, llvm::Value *> bitblock_subtract_with_borrow(llvm::Value *a, llvm::Value *b,
                                                                          llvm::Value *borrowin) {
        return bitblock_subtract_with_borrow_impl(a, b, borrowin);
    }
    // full shift producing {shiftout, shifted}
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance(llvm::Value *a, llvm::Value *shiftin, unsigned shift) {
        return bitblock_advance_impl(a, shiftin, shift);
    }
    std::pair<llvm::Value *, llvm::Value *> bitblock_indexed_advance(llvm::Value *a, llvm::Value *index_strm,
                                                                     llvm::Value *shiftin, unsigned shift) {
        return bitblock_indexed_advance_impl(a, index_strm, shiftin, shift);
    }

    llvm::Value *bitblock_mask_from(llvm::Value *const position, const bool safe = false) {
        return bitblock_mask_from_impl(position, safe);
    }
    llvm::Value *bitblock_mask_to(llvm::Value *const position, const bool safe = false) {
        return bitblock_mask_to_impl(position, safe);
    }
    llvm::Value *bitblock_set_bit(llvm::Value *const position, const bool safe = false) {
        return bitblock_set_bit_impl(position, safe);
    }

    // returns a scalar with the popcount of this block
    llvm::Value *bitblock_popcount(llvm::Value *const to_count);

    llvm::Value *simd_and(llvm::Value *a, llvm::Value *b, llvm::StringRef s = llvm::StringRef());
    llvm::Value *simd_or(llvm::Value *a, llvm::Value *b, llvm::StringRef s = llvm::StringRef());
    llvm::Value *simd_xor(llvm::Value *a, llvm::Value *b, llvm::StringRef s = llvm::StringRef());
    llvm::Value *simd_not(llvm::Value *a, llvm::StringRef s = llvm::StringRef());
    llvm::Value *fwCast(unsigned fw, llvm::Value *a);

    // Specialization interface
  public:
    virtual void CreateBaseFunctions() {}
    virtual std::string getBuilderCacheName() = 0;

  protected:
    // A bunch of places were using SmallVector<..., 16> to build up a list of indices for a shuffle. Using SmallVector
    // at all is probably premature optimization, but anyway 16 is too small: if you're dealing with AVX512 or 512-bit
    // SVE then you're going to be doing <64 x i8> shuffles. So I figure this should at least be defined one common
    // location.
    using IndexVector = llvm::SmallVector<llvm::Constant *, 64>;

    virtual llvm::Value *simd_fill_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *simd_fill_impl(unsigned vector_width, unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *simd_add_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_sub_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_mult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_eq_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_ne_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_gt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_ge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_lt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_le_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_ugt_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_ult_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_ule_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_uge_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_max_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_umax_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_min_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_umin_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_if_impl(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *simd_ternary_impl(unsigned char mask, llvm::Value *bit_2, llvm::Value *bit_1,
                                           llvm::Value *bit_0) = 0;
    virtual llvm::Value *simd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) = 0;
    virtual llvm::Value *simd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) = 0;
    virtual llvm::Value *simd_srai_impl(unsigned fw, llvm::Value *a, unsigned shift) = 0;
    virtual llvm::Value *simd_sllv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) = 0;
    virtual llvm::Value *simd_srlv_impl(unsigned fw, llvm::Value *a, llvm::Value *shifts) = 0;
    virtual llvm::Value *simd_rotl_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) = 0;
    virtual llvm::Value *simd_rotr_impl(unsigned fw, llvm::Value *a, llvm::Value *rotates) = 0;
    virtual std::vector<llvm::Value *> simd_pext_impl(unsigned fw, std::vector<llvm::Value *>,
                                                      llvm::Value *extract_mask) = 0;
    virtual llvm::Value *simd_pdep_impl(unsigned fw, llvm::Value *v, llvm::Value *deposit_mask) = 0;
    virtual llvm::Value *simd_any_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *simd_popcount_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *hsimd_partial_sum_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *simd_cttz_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *simd_bitreverse_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *esimd_mergeh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *esimd_mergel_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *esimd_bitspread_impl(unsigned vec_width, unsigned fw, llvm::Value *bitmask) = 0;
    virtual llvm::Value *hsimd_packh_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *hsimd_packl_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *hsimd_packss_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *hsimd_packus_impl(unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *hsimd_packh_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *hsimd_packl_in_lanes_impl(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b) = 0;
    virtual llvm::Value *hsimd_signmask_impl(unsigned fw, llvm::Value *a) = 0;
    virtual llvm::Value *mvmd_extract_impl(unsigned fw, llvm::Value *a, unsigned fieldIndex) = 0;
    virtual llvm::Value *mvmd_insert_impl(unsigned fw, llvm::Value *blk, llvm::Value *elt, unsigned fieldIndex) = 0;
    virtual llvm::Value *mvmd_sll_impl(unsigned fw, llvm::Value *value, llvm::Value *shift,
                                       const bool safe = false) = 0;
    virtual llvm::Value *mvmd_srl_impl(unsigned fw, llvm::Value *value, llvm::Value *shift,
                                       const bool safe = false) = 0;
    virtual llvm::Value *mvmd_slli_impl(unsigned fw, llvm::Value *a, unsigned shift) = 0;
    virtual llvm::Value *mvmd_srli_impl(unsigned fw, llvm::Value *a, unsigned shift) = 0;
    virtual llvm::Value *mvmd_dslli_impl(unsigned fw, llvm::Value *a, llvm::Value *b, unsigned shift) = 0;
    virtual llvm::Value *mvmd_dsll_impl(unsigned fw, llvm::Value *a, llvm::Value *b, llvm::Value *shift) = 0;
    virtual llvm::Value *mvmd_shuffle_impl(unsigned fw, llvm::Value *data_table, llvm::Value *index_vector,
                                           ShuffleMode m = ShuffleMode::TruncateIndex) = 0;
    virtual llvm::Value *mvmd_shuffle2_impl(unsigned fw, llvm::Value *table0, llvm::Value *table1,
                                            llvm::Value *index_vector, ShuffleMode m = ShuffleMode::TruncateIndex) = 0;
    virtual llvm::Value *mvmd_compress_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) = 0;
    virtual llvm::Value *mvmd_expand_impl(unsigned fw, llvm::Value *a, llvm::Value *select_mask) = 0;
    virtual llvm::Value *bitblock_any_impl(llvm::Value *a) = 0;
    virtual std::pair<llvm::Value *, llvm::Value *> bitblock_add_with_carry_impl(llvm::Value *a, llvm::Value *b,
                                                                                 llvm::Value *carryin) = 0;
    virtual std::pair<llvm::Value *, llvm::Value *> bitblock_subtract_with_borrow_impl(llvm::Value *a, llvm::Value *b,
                                                                                       llvm::Value *borrowin) = 0;
    virtual std::pair<llvm::Value *, llvm::Value *> bitblock_advance_impl(llvm::Value *a, llvm::Value *shiftin,
                                                                          unsigned shift) = 0;
    virtual std::pair<llvm::Value *, llvm::Value *>
    bitblock_indexed_advance_impl(llvm::Value *a, llvm::Value *index_strm, llvm::Value *shiftin, unsigned shift) = 0;
    virtual llvm::Value *bitblock_mask_from_impl(llvm::Value *const position, const bool safe = false) = 0;
    virtual llvm::Value *bitblock_mask_to_impl(llvm::Value *const position, const bool safe = false) = 0;
    virtual llvm::Value *bitblock_set_bit_impl(llvm::Value *const position, const bool safe = false) = 0;

  protected:
    [[noreturn]] void UnsupportedFieldWidthError(const unsigned FieldWidth, std::string op_name);
    llvm::Constant *bit_interleave_byteshuffle_table(unsigned fw); // support function for merge using shuffles.

    // Repeats the synthesis of ScalarF across all vector elements of aVec
    template <class ScalarF> llvm::Value *vectorize(unsigned fw, llvm::Value *aVec, ScalarF &&scalarF) {
        assert(fw >= 8);
        unsigned fieldCount = getVectorBitWidth(aVec) / fw;
        aVec = fwCast(fw, aVec); // just in case
        llvm::Value *result = llvm::PoisonValue::get(fwVectorType(fw));
        for (unsigned i = 0; i < fieldCount; i++) {
            llvm::Value *iIndex = mCB->getInt64(i);
            llvm::Value *a_i = mCB->CreateExtractElement(aVec, iIndex);
            llvm::Value *result_i = scalarF(i, a_i);
            result = mCB->CreateInsertElement(result, result_i, iIndex);
        }
        return result;
    }

    // Repeats the synthesis of ScalarF across all vector elements of aVec, bVec
    template <class ScalarF>
    llvm::Value *vectorize(unsigned fw, llvm::Value *aVec, llvm::Value *bVec, ScalarF &&scalarF) {
        assert(fw >= 8);
        unsigned fieldCount = getVectorBitWidth(aVec) / fw;
        aVec = fwCast(fw, aVec); // just in case
        bVec = fwCast(fw, bVec); // just in case
        llvm::Value *result = llvm::PoisonValue::get(fwVectorType(fw));
        for (unsigned i = 0; i < fieldCount; i++) {
            llvm::Value *iIndex = mCB->getInt64(i);
            llvm::Value *a_i = mCB->CreateExtractElement(aVec, iIndex);
            llvm::Value *b_i = mCB->CreateExtractElement(bVec, iIndex);
            llvm::Value *result_i = scalarF(i, a_i, b_i);
            result = mCB->CreateInsertElement(result, result_i, iIndex);
        }
        return result;
    }

  protected:
    CBuilder *mCB;
    unsigned mBitBlockWidth;
    unsigned mLaneWidth;
    llvm::FixedVectorType *mBitBlockType;
    llvm::Constant *mZeroInitializer;
    llvm::Constant *mOneInitializer;

    IDISA_Builder(CBuilder *cb, unsigned vectorWidth, unsigned laneWidth)
        : mCB(cb), mBitBlockWidth(vectorWidth), mLaneWidth(laneWidth),
          mBitBlockType(llvm::FixedVectorType::get(llvm::IntegerType::get(mCB->getContext(), mLaneWidth),
                                                   vectorWidth / mLaneWidth)),
          mZeroInitializer(llvm::Constant::getNullValue(mBitBlockType)),
          mOneInitializer(llvm::Constant::getAllOnesValue(mBitBlockType)) {}

    // Don't actually initialize
    IDISA_Builder(CBuilder *cb)
        : mCB(cb), mBitBlockWidth(0), mLaneWidth(0), mBitBlockType(), mZeroInitializer(), mOneInitializer() {}
};

} // namespace IDISA
