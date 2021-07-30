#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 64 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

#if defined(__ARM_ARCH)

int IDISA_ARM_Builder::arm_signmask()
{
    static const int32_t shift_ammount[] = {0,1,2,3};
    const int32x4_t shift = vld1q_s32(shift_amount);
    uint32x4_t tmp = vshrq_n_u32(uint32x4_t , 31);
    return HEDLEY_STATIC_CAST(int, vaddvq_u32(vshlq_u32(tmp, shift)));

}
Value * IDISA_ARM_Builder::hsimd_signmask(unsigned fw, Value * a) {
    if(getVectorBitWidth(a) == ARM_width)
    {
        if(fw ==32){
            Function * signmask_f64func = Intrinsic::getDeclaration(getModule(), IDISA_ARM_Builder::arm_signmask);
            Type * bitBlock_f32type = VectorType::get(getDoubleTy(), mBitBlockWidth/32);
            Value * a_as_pd = CreateBitCast(a, bitBlock_f32type);
            return CreateCall(signmask_f32func->getFunctionType(), signmask_f32func, a_as_pd);
        }
    }

 }
 #endif

// Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {

// }

// Value * IDISA_ARM_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {    

// }

// Value * IDISA_ARM_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {

// }

// Value * IDISA_ARM_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {

// }

// // full shift producing {shiftout, shifted}
// std::pair<Value *, Value *> IDISA_ARM_Builder::bitblock_advance(Value * a, Value * shiftin, unsigned shift) {

// }

// llvm::Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {

// }

// Value * IDISA_ARM_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {

// }

// Value * IDISA_ARM_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {

// }

 }
