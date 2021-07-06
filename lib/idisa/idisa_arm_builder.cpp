#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 32 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

// Value * IDISA_ARM_Builder::hsimd_signmask(unsigned fw, Value * a) {

// }

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

// }
