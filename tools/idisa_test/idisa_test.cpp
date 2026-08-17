/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <boost/filesystem.hpp>
#include <fcntl.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/core/idisa_target.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/util/hex_convert.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <sys/stat.h>
#include <toolchain/toolchain.h>
#include <vector>
namespace fs = boost::filesystem;

using namespace llvm;
using namespace kernel;

static cl::OptionCategory testFlags("Command Flags", "test options");

static cl::opt<std::string> TestOperation(cl::Positional, cl::desc("Operation to test"), cl::Required,
                                          cl::cat(testFlags));

static cl::opt<int> TestFieldWidth(cl::Positional, cl::desc("Test field width (default 64)."), cl::init(64),
                                   cl::Required, cl::cat(testFlags));

static cl::opt<std::string> Operand1TestFile(cl::Positional, cl::desc("Operand 1 data file."), cl::Required,
                                             cl::cat(testFlags));
static cl::opt<std::string> Operand2TestFile(cl::Positional, cl::desc("Operand 2 data file."), cl::Required,
                                             cl::cat(testFlags));
static cl::opt<std::string> TestOutputFile("o", cl::desc("Test output file."), cl::cat(testFlags));
static cl::opt<bool> QuietMode("q", cl::desc("Suppress output, set the return code only."), cl::cat(testFlags));
static cl::opt<int>
    ShiftMask("ShiftMask",
              cl::desc("Mask applied to the shift operand (2nd operand) of simd_sllv, srlv, srav, rotl, rotr"),
              cl::init(0));
static cl::opt<int> Immediate("i", cl::desc("Immediate value for mvmd_dslli"), cl::init(1));
static cl::opt<bool> ReportTiming("report-timing", cl::desc("Report pipeline compilation and kernel execution time"),
                                  cl::init(false), cl::cat(testFlags));

class ShiftMaskKernel : public BlockOrientedKernel {
  public:
    ShiftMaskKernel(LLVMTypeSystemInterface &ts, unsigned fw, unsigned limit, StreamSet *input, StreamSet *output);

  protected:
    void generateDoBlockMethod(KernelBuilder &kb) override;

  private:
    const unsigned mTestFw;
    const unsigned mShiftMask;
};

ShiftMaskKernel::ShiftMaskKernel(LLVMTypeSystemInterface &ts, unsigned fw, unsigned mask, StreamSet *input,
                                 StreamSet *output)
    : BlockOrientedKernel(ts, "shiftMask" + std::to_string(fw) + "_" + std::to_string(mask),
                          {Binding{"shiftOperand", input}}, {Binding{"limitedShift", output}}, {}, {}, {}),
      mTestFw(fw), mShiftMask(mask) {}

void ShiftMaskKernel::generateDoBlockMethod(KernelBuilder &b) {
    Type *fwTy = b.getIntNTy(mTestFw);
    Constant *const ZeroConst = b.getSize(0);
    Value *shiftOperand = b.loadInputStreamBlock("shiftOperand", ZeroConst);
    unsigned fieldCount = b.getBitBlockWidth() / mTestFw;
    Value *masked = b.simd_and(shiftOperand, b.getSplat(fieldCount, ConstantInt::get(fwTy, mShiftMask)));
    b.storeOutputStreamBlock("limitedShift", ZeroConst, masked);
}

struct Config {
    Type *fwTy;      // = b.getIntNTy(mTestFw)
    unsigned fw;     // = mTestFw
    unsigned fCount; // = b.getBitBlockWidth() / fw
    unsigned shift;  // = mImmediateShift
};

class IdisaOp {
  public:
    explicit IdisaOp(char const *name) : mName(name) {}
    virtual ~IdisaOp() = default;

    char const *getName() const { return mName; }

  private:
    char const *mName;
};

class IdisaUnaryOp : public IdisaOp {
  public:
    explicit IdisaUnaryOp(char const *name) : IdisaOp(name) {}
    virtual Value *createTestOp(KernelBuilder &b, Config const &c, Value *operand) const = 0;
    virtual Value *createCheckOp(KernelBuilder &b, Config const &c, Value *operand) const = 0;
};

template <class TestF, class CheckF> class GenericUnOp : public IdisaUnaryOp {
  public:
    GenericUnOp(char const *name, TestF &&testF, CheckF &&checkF)
        : IdisaUnaryOp(name), mTestF(testF), mCheckF(checkF) {}
    virtual Value *createTestOp(KernelBuilder &b, Config const &c, Value *operand) const {
        return mTestF(b, c, operand);
    }
    virtual Value *createCheckOp(KernelBuilder &b, Config const &c, Value *operand) const {
        return mCheckF(b, c, operand);
    }

  private:
    TestF mTestF;
    CheckF mCheckF;
};

class IdisaBinaryOp : public IdisaOp {
  public:
    explicit IdisaBinaryOp(char const *name) : IdisaOp(name) {}
    virtual Value *createTestOp(KernelBuilder &b, Config const &c, Value *operand1, llvm::Value *operand2) const = 0;
    virtual Value *createCheckOp(KernelBuilder &b, Config const &c, Value *operand1, llvm::Value *operand2) const = 0;
};

template <class TestF, class CheckF> class GenericBinOp : public IdisaBinaryOp {
  public:
    GenericBinOp(char const *name, TestF &&testF, CheckF &&checkF)
        : IdisaBinaryOp(name), mTestF(testF), mCheckF(checkF) {}
    virtual Value *createTestOp(KernelBuilder &b, Config const &c, Value *operand1, llvm::Value *operand2) const {
        return mTestF(b, c, operand1, operand2);
    }
    virtual Value *createCheckOp(KernelBuilder &b, Config const &c, Value *operand1, llvm::Value *operand2) const {
        return mCheckF(b, c, operand1, operand2);
    }

  private:
    TestF mTestF;
    CheckF mCheckF;
};

template <class TestF, class CheckF>
std::unique_ptr<IdisaUnaryOp> makeUnOp(char const *name, TestF &&testF, CheckF &&checkF) {
    return std::make_unique<GenericUnOp<TestF, CheckF>>(name, testF, checkF);
}

template <class HorizontalStoreCheckF> auto wrapHorizontalStoreCheck(HorizontalStoreCheckF &&horizontalStoreCheckF) {
    return [=](KernelBuilder &b, Config const &c, Value *operand1Block, llvm::Value *operand2Block) {
        unsigned fieldCount = b.getBitBlockWidth() / c.fw;
        Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
        for (unsigned i = 0; i < fieldCount; i++) {
            Value *operand1 = b.mvmd_extract(c.fw, operand1Block, i);
            Value *operand2 = b.mvmd_extract(c.fw, operand2Block, i);

            expectedBlock = horizontalStoreCheckF(b, c, operand1, operand2, i, expectedBlock);
        }
        return expectedBlock;
    };
};

template <class ScalarCheckF> auto wrapScalarCheck(ScalarCheckF &&scalarCheckF) {
    return wrapHorizontalStoreCheck([=](KernelBuilder &b, Config const &c, Value *operand1, llvm::Value *operand2,
                                        unsigned i, Value *expectedBlock) {
        Value *expected = scalarCheckF(b, c, operand1, operand2);
        return b.bitCast(b.mvmd_insert(c.fw, expectedBlock, expected, i));
    });
}

template <class TestF, class CheckF>
std::unique_ptr<IdisaBinaryOp> makeBinOp(char const *name, TestF &&testF, CheckF &&checkF) {
    return std::make_unique<GenericBinOp<TestF, CheckF>>(name, std::move(testF), std::move(checkF));
}

template <class TestF, class ScalarCheckF>
std::unique_ptr<IdisaBinaryOp> makeScalarCheckBinOp(char const *name, TestF &&testF, ScalarCheckF &&scalarCheckF) {
    return makeBinOp(name, std::move(testF), wrapScalarCheck(scalarCheckF));
}

template <class TestF, class HorizontalStoreCheckF>
std::unique_ptr<IdisaBinaryOp> makeHorizontalStoreCheckBinOp(char const *name, TestF &&testF,
                                                             HorizontalStoreCheckF &&horizontalStoreCheckF) {
    return makeBinOp(name, std::move(testF), wrapHorizontalStoreCheck(horizontalStoreCheckF));
}

std::unique_ptr<IdisaBinaryOp> binaryOps[] = {
    makeScalarCheckBinOp(
        "simd_add",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_add(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateAdd(operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_sub",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_sub(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSub(operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_mult",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_mult(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateMul(operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_eq",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_eq(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpEQ(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_ne",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_ne(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpNE(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_gt",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_gt(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpSGT(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_ugt",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_ugt(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpUGT(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_ge",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_ge(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpSGE(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_uge",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_uge(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpUGE(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_lt",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_lt(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpSLT(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_le",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_le(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpSLE(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_ult",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_ult(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpULT(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_ule",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_ule(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSExt(b.CreateICmpULE(operand1, operand2), c.fwTy);
        }),
    makeScalarCheckBinOp(
        "simd_max",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_max(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSelect(b.CreateICmpSGT(operand1, operand2), operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_min",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_min(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSelect(b.CreateICmpSLT(operand1, operand2), operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_umax",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_umax(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSelect(b.CreateICmpUGT(operand1, operand2), operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_umin",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_umin(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateSelect(b.CreateICmpULT(operand1, operand2), operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_sllv",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_sllv(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateShl(operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_srlv",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_srlv(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.CreateLShr(operand1, operand2);
        }),
    makeScalarCheckBinOp(
        "simd_rotl",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_rotl(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            Constant *fwConst = ConstantInt::get(c.fwTy, c.fw);
            Constant *fwMaskConst = ConstantInt::get(c.fwTy, c.fw - 1);
            Value *shl = b.CreateShl(operand1, b.CreateAnd(operand2, fwMaskConst));
            Value *shr = b.CreateLShr(operand1, b.CreateAnd(b.CreateSub(fwConst, operand2), fwMaskConst));
            return b.CreateOr(shl, shr);
        }),
    makeScalarCheckBinOp(
        "simd_rotr",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_rotr(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            Constant *fwConst = b.getIntN(c.fw, c.fw);
            Constant *fwMaskConst = b.getIntN(c.fw, c.fw - 1);
            Value *shl = b.CreateShl(operand1, b.CreateAnd(b.CreateSub(fwConst, operand2), fwMaskConst));
            Value *shr = b.CreateLShr(operand1, b.CreateAnd(operand2, fwMaskConst));
            return b.CreateOr(shl, shr);
        }),
    makeScalarCheckBinOp(
        "simd_pext",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_pext(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            Constant *zeroConst = b.getIntN(c.fw, 0);
            Constant *oneConst = b.getIntN(c.fw, 1);
            Value *expected = zeroConst;
            Value *out_bit = oneConst;
            for (unsigned i = 0; i < c.fw; i++) {
                Value *i_bit = b.getIntN(c.fw, 1LL << i);
                Value *operand_i_isSet = b.CreateICmpEQ(b.CreateAnd(operand1, i_bit), i_bit);
                Value *mask_i_isSet = b.CreateICmpEQ(b.CreateAnd(operand2, i_bit), i_bit);
                expected =
                    b.CreateSelect(b.CreateAnd(operand_i_isSet, mask_i_isSet), b.CreateOr(expected, out_bit), expected);
                out_bit = b.CreateSelect(mask_i_isSet, b.CreateAdd(out_bit, out_bit), out_bit);
            }
            return expected;
        }),
    makeScalarCheckBinOp(
        "simd_pdep",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.simd_pdep(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            Constant *zeroConst = ConstantInt::getNullValue(c.fwTy);
            Constant *oneConst = ConstantInt::get(c.fwTy, 1);
            Value *expected = zeroConst;
            Value *shft = zeroConst;
            Value *select_bit = oneConst;
            for (unsigned i = 0; i < c.fw; i++) {
                expected =
                    b.CreateOr(b.CreateAnd(operand2, b.CreateShl(b.CreateAnd(operand1, select_bit), shft)), expected);
                Value *i_bit = b.getIntN(c.fw, 1LL << i);
                Value *mask_i_isSet = b.CreateICmpEQ(b.CreateAnd(operand2, i_bit), i_bit);
                select_bit = b.CreateSelect(mask_i_isSet, b.CreateAdd(select_bit, select_bit), select_bit);
                shft = b.CreateSelect(mask_i_isSet, shft, b.CreateAdd(shft, oneConst));
            }
            return expected;
        }),
    makeHorizontalStoreCheckBinOp(
        "hsimd_packh",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.hsimd_packh(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2, unsigned i,
           Value *expectedBlock) -> Value * {
            operand1 = b.CreateTrunc(b.CreateLShr(operand1, c.fw / 2), b.getIntNTy(c.fw / 2));
            operand2 = b.CreateTrunc(b.CreateLShr(operand2, c.fw / 2), b.getIntNTy(c.fw / 2));
            expectedBlock = b.mvmd_insert(c.fw / 2, expectedBlock, operand1, i);
            expectedBlock = b.bitCast(b.mvmd_insert(c.fw / 2, expectedBlock, operand2, c.fCount + i));
            return expectedBlock;
        }),
    makeHorizontalStoreCheckBinOp(
        "hsimd_packl",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.hsimd_packl(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2, unsigned i,
           Value *expectedBlock) -> Value * {
            operand1 = b.CreateTrunc(operand1, b.getIntNTy(c.fw / 2));
            operand2 = b.CreateTrunc(operand2, b.getIntNTy(c.fw / 2));
            expectedBlock = b.mvmd_insert(c.fw / 2, expectedBlock, operand1, i);
            expectedBlock = b.bitCast(b.mvmd_insert(c.fw / 2, expectedBlock, operand2, c.fCount + i));
            return expectedBlock;
        }),
    makeBinOp(
        "hsimd_packus",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            operand1 = b.simd_srai(c.fw, operand1, c.fw / 2 - 1);
            operand2 = b.simd_srai(c.fw, operand2, c.fw / 2 - 1);
            return b.hsimd_packus(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            operand1Block = b.simd_srai(c.fw, operand1Block, c.fw / 2 - 1);
            operand2Block = b.simd_srai(c.fw, operand2Block, c.fw / 2 - 1);
            auto checkF = wrapHorizontalStoreCheck([=](KernelBuilder &b, Config const &c, Value *operand1,
                                                       Value *operand2, unsigned i, Value *expectedBlock) -> Value * {
                Value *zeroes = ConstantInt::getNullValue(operand1->getType());
                operand1 = b.CreateSelect(b.CreateICmpSLT(operand1, zeroes), zeroes, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSLT(operand2, zeroes), zeroes, operand2);
                Value *testVal = ConstantInt::get(b.getContext(), APInt::getLowBitsSet(c.fw, c.fw / 2));
                operand1 = b.CreateSelect(b.CreateICmpSGT(operand1, testVal), testVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSGT(operand2, testVal), testVal, operand2);
                operand1 = b.CreateTrunc(operand1, b.getIntNTy(c.fw / 2));
                operand2 = b.CreateTrunc(operand2, b.getIntNTy(c.fw / 2));
                expectedBlock = b.mvmd_insert(c.fw / 2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(c.fw / 2, expectedBlock, operand2, c.fCount + i));
                return expectedBlock;
            });
            return checkF(b, c, operand1Block, operand2Block);
        }),
    makeBinOp(
        "hsimd_packss",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            operand1 = b.simd_srai(c.fw, operand1, c.fw / 2 - 1);
            operand2 = b.simd_srai(c.fw, operand2, c.fw / 2 - 1);
            return b.hsimd_packss(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            operand1Block = b.simd_srai(c.fw, operand1Block, c.fw / 2 - 1);
            operand2Block = b.simd_srai(c.fw, operand2Block, c.fw / 2 - 1);
            auto checkF = wrapHorizontalStoreCheck([=](KernelBuilder &b, Config const &c, Value *operand1,
                                                       Value *operand2, unsigned i, Value *expectedBlock) -> Value * {
// JL -- I found 2 versions of the hsimd_packss test, the first is the one that would have been active...?
#if 1
                Value *maxVal = ConstantInt::get(b.getContext(), APInt::getLowBitsSet(c.fw, c.fw / 2 - 1));
                operand1 = b.CreateSelect(b.CreateICmpSGT(operand1, maxVal), maxVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSGT(operand2, maxVal), maxVal, operand2);
                Value *minVal = ConstantInt::get(b.getContext(), APInt::getHighBitsSet(c.fw, c.fw / 2 + 1));
                operand1 = b.CreateSelect(b.CreateICmpSLT(operand1, minVal), minVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSLT(operand2, minVal), minVal, operand2);
                operand1 = b.CreateTrunc(operand1, b.getIntNTy(c.fw / 2));
                operand2 = b.CreateTrunc(operand2, b.getIntNTy(c.fw / 2));
                expectedBlock = b.mvmd_insert(c.fw / 2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(c.fw / 2, expectedBlock, operand2, c.fCount + i));
#else
                Value *testVal = ConstantInt::get(c.fwTy, (1 << (c.fw / 2 - 1)) - 1);
                operand1 = b.CreateSelect(b.CreateICmpSGT(operand1, testVal), testVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSGT(operand2, testVal), testVal, operand2);
                testVal = b.CreateNot(testVal);
                operand1 = b.CreateSelect(b.CreateICmpSLT(operand1, testVal), testVal, operand1);
                operand2 = b.CreateSelect(b.CreateICmpSLT(operand2, testVal), testVal, operand2);
                expectedBlock = b.mvmd_insert(c.fw / 2, expectedBlock, operand1, i);
                expectedBlock = b.bitCast(b.mvmd_insert(c.fw / 2, expectedBlock, operand2, c.fCount + i));
#endif
                return expectedBlock;
            });
            return checkF(b, c, operand1Block, operand2Block);
        }),
    makeHorizontalStoreCheckBinOp(
        "esimd_mergeh",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.esimd_mergeh(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2, unsigned i,
           Value *expectedBlock) -> Value * {
            if (i >= c.fCount / 2) {
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, operand1, 2 * (i - c.fCount / 2));
                expectedBlock = b.bitCast(b.mvmd_insert(c.fw, expectedBlock, operand2, 2 * (i - c.fCount / 2) + 1));
            }
            return expectedBlock;
        }),
    makeHorizontalStoreCheckBinOp(
        "esimd_mergel",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.esimd_mergel(c.fw, operand1, operand2);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2, unsigned i,
           Value *expectedBlock) -> Value * {
            if (i < c.fCount / 2) {
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, operand1, 2 * i);
                expectedBlock = b.bitCast(b.mvmd_insert(c.fw, expectedBlock, operand2, 2 * i + 1));
            }
            return expectedBlock;
        }),
    makeBinOp(
        "mvmd_shuffle",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.mvmd_shuffle(c.fw, operand1, operand2, IDISA::IDISA_Builder::ShuffleMode::TruncateIndex);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
            for (unsigned i = 0; i < c.fCount; i++) {
                Value *idx_field = b.mvmd_extract(c.fw, operand2Block, i);
                Value *idx = b.CreateURem(idx_field, ConstantInt::get(c.fwTy, c.fCount));
                Value *elt =
                    b.CreateExtractElement(b.fwCast(c.fw, operand1Block), b.CreateZExtOrTrunc(idx, b.getInt32Ty()));
                // ShuffleMode::TruncateIndex behaviour
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, elt, i);
            }
            return expectedBlock;
        }),
    makeBinOp(
        "mvmd_shuffleO",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.mvmd_shuffle(c.fw, operand1, operand2, IDISA::IDISA_Builder::ShuffleMode::ZeroOnIndexOver);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            Constant *fieldLimit = ConstantInt::get(c.fwTy, c.fCount);
            Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
            for (unsigned i = 0; i < c.fCount; i++) {
                Value *idx_field = b.mvmd_extract(c.fw, operand2Block, i);
                Value *idx = b.CreateURem(idx_field, ConstantInt::get(c.fwTy, c.fCount));
                Value *elt =
                    b.CreateExtractElement(b.fwCast(c.fw, operand1Block), b.CreateZExtOrTrunc(idx, b.getInt32Ty()));
                // ShuffleMode::ZeroOnIndexOver behaviour
                elt = b.CreateSelect(b.CreateICmpUGE(idx_field, fieldLimit), ConstantInt::getNullValue(c.fwTy), elt);
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, elt, i);
            }
            return expectedBlock;
        }),
    makeBinOp(
        "mvmd_shuffleH",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.mvmd_shuffle(c.fw, operand1, operand2, IDISA::IDISA_Builder::ShuffleMode::ZeroOnHighIndexBit);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
            for (unsigned i = 0; i < c.fCount; i++) {
                Value *idx_field = b.mvmd_extract(c.fw, operand2Block, i);
                Value *idx = b.CreateURem(idx_field, ConstantInt::get(c.fwTy, c.fCount));
                Value *elt =
                    b.CreateExtractElement(b.fwCast(c.fw, operand1Block), b.CreateZExtOrTrunc(idx, b.getInt32Ty()));
                // ShuffleMode::ZeroOnHighIndexBit behaviour
                elt = b.CreateSelect(b.CreateICmpSLT(idx_field, ConstantInt::getNullValue(c.fwTy)),
                                     ConstantInt::getNullValue(c.fwTy), elt);
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, elt, i);
            }
            return expectedBlock;
        }),
    makeBinOp(
        "mvmd_compress",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            // Real callers (e.g. deletion.cpp) pass a scalar bitmask built with
            // hsimd_signmask, not a raw data block. Derive a realistic one from
            // operand2 here so this test actually matches production usage.
            Value *scalarMask = b.hsimd_signmask(c.fw, operand2);
            return b.mvmd_compress(c.fw, operand1, scalarMask);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            // Match the scalar bitmask built in IdisaBinaryOpTestKernel above.
            Value *scalarMask = b.hsimd_signmask(c.fw, operand2Block);
            Type *maskTy = scalarMask->getType();
            // For each input field i, precompute whether it's selected and its
            // rank (how many selected fields come before it) - this is the
            // output slot it should land in if selected.
            std::vector<Value *> isSelected(c.fCount);
            std::vector<Value *> rank(c.fCount);
            Value *runningCount = ConstantInt::get(b.getInt32Ty(), 0);
            Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
            for (unsigned i = 0; i < c.fCount; i++) {
                Value *bit =
                    b.CreateAnd(b.CreateLShr(scalarMask, ConstantInt::get(maskTy, i)), ConstantInt::get(maskTy, 1));
                isSelected[i] = b.CreateICmpNE(bit, ConstantInt::get(maskTy, 0));
                rank[i] = runningCount;
                runningCount = b.CreateAdd(runningCount, b.CreateZExt(isSelected[i], b.getInt32Ty()));
            }
            // For each output slot j, find the (at most one) selected input
            // field whose rank equals j, and place its value there.
            for (unsigned j = 0; j < c.fCount; j++) {
                Value *chosen = ConstantInt::get(c.fwTy, 0);
                for (unsigned i = 0; i < c.fCount; i++) {
                    Value *matches =
                        b.CreateAnd(isSelected[i], b.CreateICmpEQ(rank[i], ConstantInt::get(b.getInt32Ty(), j)));
                    Value *elt = b.mvmd_extract(c.fw, operand1Block, i);
                    chosen = b.CreateSelect(matches, elt, chosen);
                }
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, chosen, j);
            }
            return expectedBlock;
        }),
    makeBinOp(
        "mvmd_expand",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            Value *scalarMask = b.hsimd_signmask(c.fw, operand2);
            return b.mvmd_expand(c.fw, operand1, scalarMask);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            // Mirror image of mvmd_compress's reference above: for each output
            // slot j, if it's selected, it should hold operand1's field at
            // rank(j) (how many selected slots come before j); otherwise 0.
            Value *scalarMask = b.hsimd_signmask(c.fw, operand2Block);
            Type *maskTy = scalarMask->getType();
            std::vector<Value *> isSelected(c.fCount);
            std::vector<Value *> rank(c.fCount);
            Value *runningCount = ConstantInt::get(b.getInt32Ty(), 0);
            Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
            for (unsigned j = 0; j < c.fCount; j++) {
                Value *bit =
                    b.CreateAnd(b.CreateLShr(scalarMask, ConstantInt::get(maskTy, j)), ConstantInt::get(maskTy, 1));
                isSelected[j] = b.CreateICmpNE(bit, ConstantInt::get(maskTy, 0));
                rank[j] = runningCount;
                runningCount = b.CreateAdd(runningCount, b.CreateZExt(isSelected[j], b.getInt32Ty()));
            }
            for (unsigned j = 0; j < c.fCount; j++) {
                Value *chosen = ConstantInt::get(c.fwTy, 0);
                for (unsigned k = 0; k < c.fCount; k++) {
                    Value *matches =
                        b.CreateAnd(isSelected[j], b.CreateICmpEQ(rank[j], ConstantInt::get(b.getInt32Ty(), k)));
                    Value *elt = b.mvmd_extract(c.fw, operand1Block, k);
                    chosen = b.CreateSelect(matches, elt, chosen);
                }
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, chosen, j);
            }
            return expectedBlock;
        }),
    makeBinOp(
        "mvmd_dslli",
        [](KernelBuilder &b, Config const &c, Value *operand1, Value *operand2) -> Value * {
            return b.mvmd_dslli(c.fw, operand1, operand2, c.shift);
        },
        [](KernelBuilder &b, Config const &c, Value *operand1Block, Value *operand2Block) -> Value * {
            Value *expectedBlock = Constant::getNullValue(b.fwVectorType(c.fw));
            for (unsigned i = 0; i < c.fCount; i++) {
                Value *elt = nullptr;
                if (i < c.shift)
                    elt = b.mvmd_extract(c.fw, operand2Block, c.fCount - c.shift + i);
                else
                    elt = b.mvmd_extract(c.fw, operand1Block, i - c.shift);
                expectedBlock = b.mvmd_insert(c.fw, expectedBlock, elt, i);
            }
            return expectedBlock;
        }),
};

// static cl::opt<IDISA::IDISA_Builder::ShuffleMode>
//     ShuffleIndex("ShuffleIndex",
//                  cl::values(clEnumValN(IDISA::IDISA_Builder::ShuffleMode::TruncateIndex, "Truncate",
//                                        "Truncate out-of-bound shuffle indexes."),
//                             clEnumValN(IDISA::IDISA_Builder::ShuffleMode::ZeroOnIndexOver, "ZeroOnOver",
//                                        "Select zero for shuffle indexes out of bound."),
//                             clEnumValN(IDISA::IDISA_Builder::ShuffleMode::ZeroOnHighIndexBit, "ZeroOnHighBit",
//                                        "Select zero if high index bit set, otherwise truncate.")),
//                  cl::init(IDISA::IDISA_Builder::ShuffleMode::TruncateIndex));

class IdisaBinaryOpTestKernel : public MultiBlockKernel {
  public:
    IdisaBinaryOpTestKernel(LLVMTypeSystemInterface &ts, const IdisaBinaryOp *idisa_op, unsigned fw, unsigned imm,
                            StreamSet *Operand1, StreamSet *Operand2, StreamSet *result);

  protected:
    void generateMultiBlockLogic(KernelBuilder &kb, llvm::Value *const numOfStrides) override;

  private:
    const IdisaBinaryOp *mIdisaOperation;
    const unsigned mTestFw;
    const unsigned mImmediateShift;
};

IdisaBinaryOpTestKernel::IdisaBinaryOpTestKernel(LLVMTypeSystemInterface &ts, const IdisaBinaryOp *idisa_op,
                                                 unsigned fw, unsigned imm, StreamSet *Operand1, StreamSet *Operand2,
                                                 StreamSet *result)
    : MultiBlockKernel(ts, idisa_op->getName() + std::to_string(fw) + "_test",
                       {Binding{"operand1", Operand1}, Binding{"operand2", Operand2}}, {Binding{"result", result}}, {},
                       {}, {}),
      mIdisaOperation(idisa_op), mTestFw(fw), mImmediateShift(imm) {}

void IdisaBinaryOpTestKernel::generateMultiBlockLogic(KernelBuilder &b, llvm::Value *const numOfBlocks) {
    BasicBlock *entry = b.GetInsertBlock();
    BasicBlock *processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock *done = b.CreateBasicBlock("done");
    Constant *const ZeroConst = b.getSize(0);
    b.CreateBr(processBlock);
    b.SetInsertPoint(processBlock);
    PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZeroConst, entry);
    Value *operand1 = b.loadInputStreamBlock("operand1", ZeroConst, blockOffsetPhi);
    Value *operand2 = b.loadInputStreamBlock("operand2", ZeroConst, blockOffsetPhi);

    Config c = {
        /*.fwTy =*/b.getIntNTy(mTestFw),
        /*.fw =*/mTestFw,
        /*.fCount =*/b.getBitBlockWidth() / mTestFw,
        /*.shift =*/mImmediateShift,
    };
    Value *result = mIdisaOperation->createTestOp(b, c, operand1, operand2);

    b.storeOutputStreamBlock("result", ZeroConst, blockOffsetPhi, b.bitCast(result));
    Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, processBlock);
    Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, processBlock, done);
    b.SetInsertPoint(done);
}

class IdisaBinaryOpCheckKernel : public BlockOrientedKernel {
  public:
    IdisaBinaryOpCheckKernel(LLVMTypeSystemInterface &ts, IdisaBinaryOp const *op, unsigned fw, unsigned imm,
                             StreamSet *Operand1, StreamSet *Operand2, StreamSet *result, StreamSet *expected,
                             Scalar *failures);

  protected:
    void generateDoBlockMethod(KernelBuilder &kb) override;

  private:
    const IdisaBinaryOp *mIdisaOperation;
    const unsigned mTestFw;
    const unsigned mImmediateShift;
};

IdisaBinaryOpCheckKernel::IdisaBinaryOpCheckKernel(LLVMTypeSystemInterface &ts, const IdisaBinaryOp *op, unsigned fw,
                                                   unsigned imm, StreamSet *Operand1, StreamSet *Operand2,
                                                   StreamSet *result, StreamSet *expected, Scalar *failures)
    : BlockOrientedKernel(
          ts, op->getName() + std::to_string(fw) + "_check" + std::to_string(QuietMode),
          {Binding{"operand1", Operand1}, Binding{"operand2", Operand2}, Binding{"test_result", result}},
          {Binding{"expected_result", expected}}, {}, {Binding{"totalFailures", failures}}, {}),
      mIdisaOperation(op), mTestFw(fw), mImmediateShift(imm) {}

void IdisaBinaryOpCheckKernel::generateDoBlockMethod(KernelBuilder &b) {
    Constant *const ZeroConst = b.getSize(0);
    Value *operand1Block = b.loadInputStreamBlock("operand1", ZeroConst);
    Value *operand2Block = b.loadInputStreamBlock("operand2", ZeroConst);
    Value *resultBlock = b.loadInputStreamBlock("test_result", ZeroConst);

    Config c = {
        /*.fwTy =*/b.getIntNTy(mTestFw),
        /*.fw =*/mTestFw,
        /*.fCount =*/b.getBitBlockWidth() / mTestFw,
        /*.shift =*/mImmediateShift,
    };
    Value *expectedBlock = mIdisaOperation->createCheckOp(b, c, operand1Block, operand2Block);

    b.storeOutputStreamBlock("expected_result", ZeroConst, expectedBlock);
    Value *failures = b.simd_ugt(mTestFw, b.simd_xor(resultBlock, expectedBlock), b.allZeroes());
    Value *anyFailure = b.bitblock_any(failures);
    Value *failure_count = b.CreateUDiv(b.bitblock_popcount(failures), b.getSize(mTestFw));
    b.setScalarField("totalFailures", b.CreateAdd(b.getScalarField("totalFailures"), failure_count));
    if (!QuietMode) {
        // created here so they always get terminators; unterminated blocks crash the JIT under -q
        BasicBlock *reportFailure = b.CreateBasicBlock("reportFailure");
        BasicBlock *continueTest = b.CreateBasicBlock("continueTest");
        b.CreateCondBr(anyFailure, reportFailure, continueTest);
        b.SetInsertPoint(reportFailure);
        b.CallPrintRegister("operand1", b.bitCast(operand1Block));
        b.CallPrintRegister("operand2", b.bitCast(operand2Block));
        b.CallPrintRegister(std::string(mIdisaOperation->getName()) + "(" + std::to_string(mTestFw) +
                                ", operand1, operand2)",
                            resultBlock);
        b.CallPrintRegister("expecting", expectedBlock);
        b.CreateBr(continueTest);
        b.SetInsertPoint(continueTest);
    }
}

// Open a file and return its file desciptor.
int32_t openFile(const std::string &fileName, llvm::raw_ostream &msgstrm) {
    if (fileName == "-") {
        return STDIN_FILENO;
    } else {
        struct stat sb;
        int32_t fileDescriptor = open(fileName.c_str(), O_RDONLY);
        if (LLVM_UNLIKELY(fileDescriptor == -1)) {
            if (errno == EACCES) {
                msgstrm << "idisa_test: " << fileName << ": Permission denied.\n";
            } else if (errno == ENOENT) {
                msgstrm << "idisa_test: " << fileName << ": No such file.\n";
            } else {
                msgstrm << "idisa_test: " << fileName << ": Failed.\n";
            }
            return fileDescriptor;
        }
        if (stat(fileName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            msgstrm << "idisa_test: " << fileName << ": Is a directory.\n";
            close(fileDescriptor);
            return -1;
        }
        return fileDescriptor;
    }
}

typedef size_t (*IDISAtestFunctionType)(uint32_t fd1, uint32_t fd2, const char *outputFileName);

inline StreamSet *readHexToBinary(PipelineBuilder &P, const std::string &fd) {
    StreamSet *const hexStream = P.CreateStreamSet(1, 8);
    Scalar *const fileDecriptor = P.getInputScalar(fd);
    P.CreateKernelCall<ReadSourceKernel>(fileDecriptor, hexStream);
    StreamSet *const bitStream = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<HexToBinary>(hexStream, bitStream);
    return bitStream;
}

inline StreamSet *applyShiftMask(kernel::PipelineBuilder &P, StreamSet *input) {
    if (ShiftMask > 0) {
        StreamSet *output = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<ShiftMaskKernel>(TestFieldWidth, ShiftMask, input, output);
        return output;
    }
    return input;
}

IDISAtestFunctionType pipelineGen(CPUDriver &driver) {

    auto P = CreatePipeline(driver, Input<uint32_t>{"operand1FileDecriptor"}, Input<uint32_t>{"operand2FileDecriptor"},
                            Input<const char *>{"outputFileName"}, Output<size_t>{"totalFailures"});

    StreamSet *Operand1BitStream = readHexToBinary(P, "operand1FileDecriptor");
    StreamSet *Operand2BitStream = applyShiftMask(P, readHexToBinary(P, "operand2FileDecriptor"));

    StreamSet *ResultBitStream = P.CreateStreamSet(1, 1);

    llvm::StringRef opName = TestOperation;
    IdisaBinaryOp *testOp = nullptr;
    for (auto const &op : binaryOps) {
        if (opName == op->getName()) {
            testOp = op.get();
            break;
        }
    }
    if (!testOp) {
        llvm::report_fatal_error(llvm::StringRef("Binary operation ") + opName +
                                 " is unknown to the IdisaBinaryOpTestKernel kernel.");
    }

    P.CreateKernelCall<IdisaBinaryOpTestKernel>(testOp, TestFieldWidth, Immediate, Operand1BitStream, Operand2BitStream,
                                                ResultBitStream);

    StreamSet *ExpectedResultBitStream = P.CreateStreamSet(1, 1);

    P.CreateKernelCall<IdisaBinaryOpCheckKernel>(testOp, TestFieldWidth, Immediate, Operand1BitStream,
                                                 Operand2BitStream, ResultBitStream, ExpectedResultBitStream,
                                                 P.getOutputScalar("totalFailures"));

    if (!TestOutputFile.empty()) {
        StreamSet *ResultHexStream = P.CreateStreamSet(1, 8);
        P.CreateKernelCall<BinaryToHex>(ResultBitStream, ResultHexStream);
        Scalar *outputFileName = P.getInputScalar("outputFileName");
        P.CreateKernelCall<FileSink>(outputFileName, ResultHexStream);
    }

    return P.compile();
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&testFlags, codegen::codegen_flags()});
    CPUDriver driver("idisa_test");
    // only shift ops need the operand2 limit; elsewhere it strips sign bits the tests need
    const bool isShiftOp = TestOperation == "simd_sllv" || TestOperation == "simd_srlv" ||
                           TestOperation == "simd_rotl" || TestOperation == "simd_rotr";
    if (ShiftMask == 0 && isShiftOp) {
        ShiftMask = TestFieldWidth - 1;
    }

    std::chrono::steady_clock::time_point compileStart, compileEnd;
    if (ReportTiming)
        compileStart = std::chrono::steady_clock::now();
    auto idisaTestFunction = pipelineGen(driver);
    if (ReportTiming)
        compileEnd = std::chrono::steady_clock::now();

    const int32_t fd1 = openFile(Operand1TestFile, llvm::outs());
    const int32_t fd2 = openFile(Operand2TestFile, llvm::outs());

    std::chrono::steady_clock::time_point execStart, execEnd;
    if (ReportTiming)
        execStart = std::chrono::steady_clock::now();
    const size_t failure_count = idisaTestFunction(fd1, fd2, TestOutputFile.ValueStr.data());
    if (ReportTiming)
        execEnd = std::chrono::steady_clock::now();

    if (!QuietMode) {
        if (failure_count == 0) {
            llvm::outs() << "Test success: " << TestOperation << "<" << TestFieldWidth << ">\n";
        } else {
            llvm::outs() << "Test failure: " << TestOperation << "<" << TestFieldWidth << "> failed " << failure_count
                         << " tests!\n";
        }
    }
    if (ReportTiming) {
        const auto compileUs = std::chrono::duration_cast<std::chrono::microseconds>(compileEnd - compileStart).count();
        const auto execUs = std::chrono::duration_cast<std::chrono::microseconds>(execEnd - execStart).count();
        llvm::outs() << "Pipeline compile time: " << compileUs << " us\n";
        llvm::outs() << "Kernel execution time: " << execUs << " us\n";
    }

    close(fd1);
    close(fd2);
    return failure_count > 0;
}
