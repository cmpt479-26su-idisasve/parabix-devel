#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 64 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

/*
        if (fw == 64) {
            Function * signmask_f64func = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_sse2_movmsk_pd);
            Type * bitBlock_f64type = VectorType::get(getDoubleTy(), mBitBlockWidth/64);
            Value * a_as_pd = CreateBitCast(a, bitBlock_f64type);
            return CreateCall(signmask_f64func->getFunctionType(), signmask_f64func, a_as_pd);
        }
*/

/* Creates a call to neon_vldq to shift 4 vector */
Value* IDISA_ARM_Builder::neon_vld1x4() {
  // create a function* to the vld1x4
  Function* shiftl_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vld1x4);
  // create a static vector of {0,1,2,3} that represents the shift amounts
  static const std::vector<Value*> shuffle_amount = {
        ConstantInt::get(getInt32Ty(), 0),
        ConstantInt::get(getInt32Ty(), 1),
        ConstantInt::get(getInt32Ty(), 2),
        ConstantInt::get(getInt32Ty(), 3)
    };
  // create a function call
  return CreateCall(shiftl_f32func->getFunctionType(), shiftl_f32func, shuffle_amount);
}

/* shifts the value to the right by 31 bit to get the most sig bits */ 
Value* IDISA_ARM_Builder::neon_shrq(Value* a) {
  Function* shiftr_n_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vrshiftn);
  Type* bitBlock_f32type = VectorType::get(getInt32Ty(), mBitBlockWidth/32);
  Value* a_as_pd = CreateBitCast(a, bitBlock_f32type);
  return CreateCall(shiftr_n_f32func->getFunctionType(), shiftr_n_f32func, {a_as_pd, ConstantInt::get(getInt32Ty(), 31)});
}

/* shift lefts everything */
Value* IDISA_ARM_Builder::neon_shlq(Value* a, Value* b) {
  Function* shiftl_u_f32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vshiftu);
  return CreateCall(shiftl_u_f32func->getFunctionType(), shiftl_u_f32func, {a, b});
}

Value* IDISA_ARM_Builder::hsimd_signmask(unsigned fw, Value* a) {
    if (getVectorBitWidth(a) == ARM_width) {
        if(fw == 32) {
          /* shift operation */
          Value* shift = neon_vld1x4();
          Value* temp = neon_shrq(a);
          Value* shift_left_a = neon_shlq(temp, shift); 

          /* finally add it to the vector */
          Function* add_32func = Intrinsic::getDeclaration(getModule(), Intrinsic::arm_neon_vqaddu);
          return CreateCall(add_32func->getFunctionType(), add_32func,shift_left_a);

          /*
          SIMDE implementation
          static const int32_t shift_amount[] = {0,1,2,3};
          const int32x4_t shift = vld1q_s32(shift_amount);
          uint32x4_t tmp = vshrq_n_u32(a, 31);
          return HEDLEY_STATIC_CAST(LLVM::Value*, vaddvq_u32(vshlq_u32(tmp, shift)));
          */
        }
    }
}

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
