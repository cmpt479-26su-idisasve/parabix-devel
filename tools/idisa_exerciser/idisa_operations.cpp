/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "idisa_operations.h"

#include <kernel/io/source_kernel.h>
#include <kernel/util/hex_convert.h>

#include "idisa_exerciser.h"

using namespace std;
using namespace llvm;
using namespace kernel;

//////////////////////////////////////////////////////////////////////////////////
// ShiftMaskKernel, used potentially by the config

// class ShiftMaskKernel : public BlockOrientedKernel {
//   public:
//     ShiftMaskKernel(LLVMTypeSystemInterface &ts, unsigned fw, unsigned limit, StreamSet *input, StreamSet *output);

//   protected:
//     void generateDoBlockMethod(KernelBuilder &kb) override;

//   private:
//     const unsigned mTestFw;
//     const unsigned mShiftMask;
// };

// ShiftMaskKernel::ShiftMaskKernel(LLVMTypeSystemInterface &ts, unsigned fw, unsigned mask, StreamSet *input,
//                                  StreamSet *output)
//     : BlockOrientedKernel(ts, "shiftMask" + std::to_string(fw) + "_" + std::to_string(mask),
//                           {Binding{"shiftOperand", input}}, {Binding{"limitedShift", output}}, {}, {}, {}),
//       mTestFw(fw), mShiftMask(mask) {}

// void ShiftMaskKernel::generateDoBlockMethod(KernelBuilder &b) {
//     Type *fwTy = b.getIntNTy(mTestFw);
//     Constant *const ZeroConst = b.getSize(0);
//     Value *shiftOperand = b.loadInputStreamBlock("shiftOperand", ZeroConst);
//     unsigned fieldCount = b.getBitBlockWidth() / mTestFw;
//     Value *masked = b.simd_and(shiftOperand, b.getSplat(fieldCount, ConstantInt::get(fwTy, mShiftMask)));
//     b.storeOutputStreamBlock("limitedShift", ZeroConst, masked);
// }

//////////////////////////////////////////////////////////////////////////////////
// Kernels used by the OperationConfigs

class TestKernel : public MultiBlockKernel {
  public:
    TestKernel(LLVMTypeSystemInterface &ts, OperationConfig &config, const vector<StreamSet *> &operandSSs,
               StreamSet *testOutput)
        : MultiBlockKernel(ts, "test_" + config.getIdentifier().str(), captureOperandBindings(operandSSs),
                           {{"test_output", testOutput}}, {}, {}, {}),
          mConfig(config), mNumOperands(operandSSs.size()) {}

  protected:
    void generateMultiBlockLogic(KernelBuilder &b, llvm::Value *const numBlocks) override {
        mConfig.logKernelBuilder(b);

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *processBlock = b.CreateBasicBlock("processBlock");
        BasicBlock *done = b.CreateBasicBlock("done");
        b.CreateBr(processBlock);
        b.SetInsertPoint(processBlock);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        Constant *zero = b.getSize(0);
        blockOffsetPhi->addIncoming(zero, entry);

        vector<Value *> operandBlocks;
        for (unsigned i = 0; i < mNumOperands; ++i) {
            operandBlocks.emplace_back(b.loadInputStreamBlock(OperationConfig::operandIdent(i), zero, blockOffsetPhi));
        }
        Value *testOutputBlock = mConfig.makeTestLogic(b, operandBlocks);

        b.storeOutputStreamBlock(OperationConfig::testOutputIdent, zero, blockOffsetPhi, b.bitCast(testOutputBlock));
        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, processBlock);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numBlocks);
        b.CreateCondBr(moreToDo, processBlock, done);
        b.SetInsertPoint(done);
    }

  private:
    OperationConfig &mConfig;
    unsigned mNumOperands;

    Bindings captureOperandBindings(const vector<StreamSet *> &operandSSs) {
        Bindings bs;
        bs.reserve(1);
        unsigned i = 0;
        for (auto ss : operandSSs) {
            string name = OperationConfig::operandIdent(i++);
            bs.emplace_back(name, ss);
        }
        return bs;
    }

    TestKernel(const TestKernel &) = delete;
};

class CheckKernel : public BlockOrientedKernel {
  public:
    CheckKernel(LLVMTypeSystemInterface &ts, OperationConfig &config, bool quiet, const vector<StreamSet *> &operandSSs,
                StreamSet *testOutput, StreamSet *expectedOutput, Scalar *failureCount)
        : BlockOrientedKernel(ts, "check_" + config.getIdentifier().str(),
                              captureOperandBindings(operandSSs, testOutput),
                              {Binding{OperationConfig::expectedOutputIdent, expectedOutput}}, {},
                              {Binding{OperationConfig::failureCountIdent, failureCount}}, {}),
          mConfig(config), mQuiet(quiet), mNumOperands(operandSSs.size()) {}

  protected:
    void generateDoBlockMethod(KernelBuilder &b) override {
        Constant *zero = b.getSize(0);
        Value *testOutputBlock = b.loadInputStreamBlock(OperationConfig::testOutputIdent, zero);
        vector<Value *> operandBlocks;
        for (unsigned i = 0; i < mNumOperands; ++i) {
            operandBlocks.emplace_back(b.loadInputStreamBlock(OperationConfig::operandIdent(i), zero));
        }
        Value *expectedOutputBlock = mConfig.makeCheckLogic(b, operandBlocks);

        b.storeOutputStreamBlock(OperationConfig::expectedOutputIdent, zero, expectedOutputBlock);
        Value *failures =
            b.simd_ugt(mConfig.getFieldWidth(), b.simd_xor(testOutputBlock, expectedOutputBlock), b.allZeroes());
        Value *failureCount = b.CreateUDiv(b.bitblock_popcount(failures), b.getSize(mConfig.getFieldWidth()));
        b.setScalarField(OperationConfig::failureCountIdent,
                         b.CreateAdd(b.getScalarField(OperationConfig::failureCountIdent), failureCount));
        if (!mQuiet) {
            // created here so they always get terminators; unterminated blocks crash the JIT under -q
            BasicBlock *reportFailure = b.CreateBasicBlock("reportFailure");
            BasicBlock *continueTest = b.CreateBasicBlock("continueTest");
            Value *anyFailure = b.bitblock_any(failures);
            b.CreateCondBr(anyFailure, reportFailure, continueTest);
            b.SetInsertPoint(reportFailure);
            for (unsigned i = 0; i < mNumOperands; ++i) {
                b.CallPrintRegister(OperationConfig::operandIdent(i), b.bitCast(operandBlocks[i]));
            }
            b.CallPrintRegister(mConfig.getDescription(), testOutputBlock);
            b.CallPrintRegister("expecting", b.bitCast(expectedOutputBlock));
            b.CreateBr(continueTest);
            b.SetInsertPoint(continueTest);
        }
    }

  private:
    OperationConfig &mConfig;
    bool mQuiet;
    unsigned mNumOperands;

    Bindings captureOperandBindings(const vector<StreamSet *> &operandSSs, StreamSet *testOutput) {
        Bindings bs;
        unsigned i = 0;
        for (auto ss : operandSSs) {
            bs.emplace_back(OperationConfig::operandIdent(i++), ss);
        }
        bs.emplace_back(OperationConfig::testOutputIdent, testOutput);
        return bs;
    }
};

class DummyCheckKernel : public Kernel {
  public:
    DummyCheckKernel(LLVMTypeSystemInterface &ts, OperationConfig &config, StreamSet *testOutput, Scalar *failureCount)
        : Kernel(ts, TypeId::SegmentOriented, "dummycheck_" + config.getIdentifier().str(),
                 {Binding(OperationConfig::testOutputIdent, testOutput)}, {}, {},
                 {Binding{OperationConfig::failureCountIdent, failureCount}}, {}) {}

  protected:
    void generateKernelMethod(KernelBuilder &b) override {}

    void generateFinalizeMethod(KernelBuilder &b) override {
        b.setScalarField(OperationConfig::failureCountIdent, b.getSize(0));
    }
};

//////////////////////////////////////////////////////////////////////////////////
// OperationConfig implementation

OperationConfig::~OperationConfig() {
    for (auto const &f : mDtorFuncs) {
        f();
    }
}

void OperationConfig::fillParams(KernelBuilder &b, vector<Value *> operands, Params &outParams) const {
    outParams.fw = mFieldWidth;
    outParams.fn = b.getBitBlockWidth() / mFieldWidth;
    outParams.fTy = b.getIntNTy(mFieldWidth);
    outParams.vTy = b.fwVectorType(mFieldWidth);
    outParams.i32Ty = b.getInt32Ty();
}

void OperationConfig::resetPipeline() {
    for (auto const &f : mResetFuncs) {
        f();
    }
}

bool OperationConfig::configurePipelineFromArgs(cl::list<string> &args, bool doChecks) {
    assert(mPipelineBuilder);
#ifndef NDEBUG
    assert(!mPipelineConfigured);
    mPipelineConfigured = true;
#endif

    size_t cur = 0;
    if (parseRelevantArgs(args, cur)) {
        return true;
    }
    if (cur < args.size()) {
        return args.error("Unexpected operation arguments: expected " + to_string(cur) + ", got " + to_string(cur));
    }

    return false;
}

bool OperationConfig::parseRelevantArgs(cl::list<string> &args, size_t &cur) { return false; }

bool OperationConfig::parseUnsigned(cl::list<string> &args, size_t &cur, StringRef expectName, unsigned minVal,
                                    unsigned maxVal, unsigned &outResult) {
    if (cur >= args.size()) {
        return args.error("Not enough arguments, expected " + expectName);
    }
    string argStr = args[cur];
    char *pc = &argStr[0];
    long long result = strtoll(pc, &pc, 10);
    if (*pc != 0) {
        return args.error("Expected " + expectName + "; could not parse value");
    }
    if ((result < static_cast<long long>(minVal)) || (result > static_cast<long long>(maxVal))) {
        return args.error("Expected " + expectName + "; value outside range " + to_string(minVal) + "-" +
                          to_string(maxVal));
    }
    ++cur;
    return false;
}

bool OperationConfig::parseStreamSource(cl::list<string> &args, size_t &cur, StringRef expectName, StringRef fdIdent,
                                        int32_t &outFDResult, StreamSet *&outSSResult) {
    // Is it a filename? For now assume it is.
    // Future work:
    // - syntax to specify internal generators that generate into memory such as
    //   - random numbers w/seed?
    //   - structured test patterns?
    //   - patterns from an internal library?
    //   - synthesized fuzzing patterns?
    // - syntax to specify processing chains (hex parsing, shift mask)
    if (cur >= args.size()) {
        return args.error("Not enough arguments, expected " + expectName);
    }
    if (openInputFile(args, args[cur], outFDResult)) {
        return true;
    }
    if (outFDResult == STDIN_FILENO) {
        if (mStdinGrabbed) {
            return args.error("At most 1 input can come from STDIN");
        }
        mStdinGrabbed = true;
    }
    ++cur;

    Scalar *fileDescriptor = mPipelineBuilder->getInputScalar(fdIdent);
    assert(fileDescriptor);
    StreamSet *bitStream = mPipelineBuilder->CreateStreamSet(1, 8);
    mPipelineBuilder->CreateKernelCall<ReadSourceKernel>(fileDescriptor, bitStream);
    // Uncomment below (and comment above) to get back the default behaviour of parsing the input as a hex file
    // -- I would much prefer if this was associated with some kind of syntax.
    // Then we might consider having ways to specify different processing chains...
    // Scalar *fileDescriptor = mPipelineBuilder->getInputScalar(outFDResult);
    // StreamSet * hexStream = mPipelineBuilder->CreateStreamSet(1, 8);
    // StreamSet *bitStream = mPipelineBuilder->CreateStreamSet(1, 1);
    // mPipelineBuilder->CreateKernelCall<ReadSourceKernel>(fileDescriptor, hexStream);
    // mPipelineBuilder->CreateKernelCall<HexToBinary>(hexStream, bitStream);
    outSSResult = bitStream;

    // Uncomment (and fix names) to get shift masked input...
    // if (ShiftMask > 0) {
    //     StreamSet *shiftedStream = P.CreateStreamSet(1, 1);
    //     P.CreateKernelCall<ShiftMaskKernel>(OperationFieldWidth, ShiftMask, stream, shiftedStream);
    //     stream = shiftedStream;
    // }

    return false;
}

bool OperationConfig::openInputFile(cl::Option &arg, StringRef fileName, int32_t &outFDResult) {
    if (fileName == "-") {
        outFDResult = STDIN_FILENO;
        return false;
    } else {
        int32_t fileDescriptor = open(fileName.str().c_str(), O_RDONLY);
        if (fileDescriptor == -1) {
            return arg.error(fileName + ": " + strerror(errno));
        }
        char temp;
        if ((read(fileDescriptor, &temp, 0) == -1) && (errno == EISDIR)) {
            close(fileDescriptor);
            return arg.error(fileName + ": Is a directory");
        }
        mResetFuncs.emplace_back([=]() { lseek(fileDescriptor, 0, SEEK_SET); });
        mDtorFuncs.emplace_back([=]() { close(fileDescriptor); });
        outFDResult = fileDescriptor;
        return false;
    }
}

//////////////////////////////////////////////////////////////////////////////////
// Specializations of config to handle different classes of operations

template <unsigned N> class NaryOpConfig : public OperationConfig {
  public:
    struct Params : public OperationConfig::Params {
        Value *opr[N]; // operands
    };

    explicit NaryOpConfig(StringRef description, StringRef identifier, unsigned fieldWidth)
        : OperationConfig(description, identifier, fieldWidth) {}

    void fillParams(KernelBuilder &b, std::vector<Value *> operands, Params &outParams) const {
        assert(operands.size() == N);
        OperationConfig::fillParams(b, operands, outParams);
        for (unsigned i = 0; i < N; ++i) {
            outParams.opr[i] = operands[i];
        }
    }

  protected:
    vector<int32_t> mOperandFDs;
    function<void()> mCompileFunc;

    void constructPipeline(CPUDriver &driver) override {
        assert(!mPipelineBuilder);
        // I found lots of template magic in the PipelineBuilder, but nothing that would assist with building up a
        // function signature and matching call programmatically. Since we don't actually need a fully general system,
        // this mediocrity will do: at least it keeps the special-casing local, so it can be replaced easily if a better
        // solution is wanted.
        switch (N) {
        case 0: {
            auto pTmp = CreatePipeline(driver, Input<const char *>{OperationConfig::outputFilenameIdent},
                                       Output<size_t>{OperationConfig::failureCountIdent});
            auto p = make_unique<decltype(pTmp)>(move(pTmp));
            auto pPtr = p.get();
            mPipelineBuilder = move(p);
            mCompileFunc = [=, this]() {
                auto pipelineFunc = pPtr->compile();
                mExecuteFunc = [=, this]() -> size_t { return pipelineFunc(mOutputFilename.c_str()); };
            };
        } break;
        case 1: {
            auto pTmp = CreatePipeline(driver, Input<int32_t>{OperationConfig::operandFDIdent(0)},
                                       Input<const char *>{OperationConfig::outputFilenameIdent},
                                       Output<size_t>{OperationConfig::failureCountIdent});
            auto p = make_unique<decltype(pTmp)>(move(pTmp));
            auto pPtr = p.get();
            mPipelineBuilder = move(p);
            mCompileFunc = [=, this]() {
                auto pipelineFunc = pPtr->compile();
                mExecuteFunc = [=, this]() -> size_t { return pipelineFunc(mOperandFDs[0], mOutputFilename.c_str()); };
            };
        } break;
        case 2: {
            auto pTmp = CreatePipeline(driver, Input<int32_t>{OperationConfig::operandFDIdent(0)},
                                       Input<int32_t>{OperationConfig::operandFDIdent(1)},
                                       Input<const char *>{OperationConfig::outputFilenameIdent},
                                       Output<size_t>{OperationConfig::failureCountIdent});
            auto p = make_unique<decltype(pTmp)>(move(pTmp));
            auto pPtr = p.get();
            mPipelineBuilder = move(p);
            mCompileFunc = [=, this]() {
                auto pipelineFunc = pPtr->compile();
                mExecuteFunc = [=, this]() -> size_t {
                    return pipelineFunc(mOperandFDs[0], mOperandFDs[1], mOutputFilename.c_str());
                };
            };
        } break;
        case 3: {
            auto pTmp = CreatePipeline(driver, Input<int32_t>{OperationConfig::operandFDIdent(0)},
                                       Input<int32_t>{OperationConfig::operandFDIdent(1)},
                                       Input<int32_t>{OperationConfig::operandFDIdent(2)},
                                       Input<const char *>{OperationConfig::outputFilenameIdent},
                                       Output<size_t>{OperationConfig::failureCountIdent});
            auto p = make_unique<decltype(pTmp)>(move(pTmp));
            auto pPtr = p.get();
            mPipelineBuilder = move(p);
            mCompileFunc = [=, this]() {
                auto pipelineFunc = pPtr->compile();
                mExecuteFunc = [=, this]() -> size_t {
                    return pipelineFunc(mOperandFDs[0], mOperandFDs[1], mOperandFDs[2], mOutputFilename.c_str());
                };
            };
        } break;
        default:
            assert(!"Unimplemented N > 3");
        }
    }

    void compilePipeline() {
        assert(mPipelineBuilder);
        mCompileFunc();
        assert(!mKernelBuilderID.empty());
    }

    size_t executePipeline() {
        resetPipeline();
        assert(mPipelineBuilder);
        return mExecuteFunc();
    }

    bool configurePipelineFromArgs(cl::list<string> &args, bool doChecks) override {
        if (OperationConfig::configurePipelineFromArgs(args, doChecks)) {
            return true;
        }

        assert(mOperandSSs.size() == N);
        return false;
    }

    bool parseRelevantArgs(cl::list<string> &args, size_t &cur) override {
        if (OperationConfig::parseRelevantArgs(args, cur)) {
            return true;
        }

        for (unsigned i = 0; i < N; ++i) {
            int32_t fd = -1;
            StreamSet *ss = nullptr;
            if (parseStreamSource(args, cur, "operand " + to_string(i + 1), OperationConfig::operandFDIdent(i), fd,
                                  ss)) {
                return true;
            }
            mOperandFDs.emplace_back(fd);
            mOperandSSs.emplace_back(ss);
        }
        return false;
    }
};

template <class BaseOpConfig, unsigned MinVal = 0, unsigned MaxVal = UINT_MAX>
class ImmediateOpConfig : public BaseOpConfig {
  public:
    struct Params : public BaseOpConfig::Params {
        uint64_t immed; // immediate
    };

    explicit ImmediateOpConfig(StringRef description, StringRef identifier, unsigned fieldWidth)
        : BaseOpConfig(description, identifier, fieldWidth), mImmediateValue(0) {}

    void fillParams(KernelBuilder &b, std::vector<Value *> operands, Params &outParams) const {
        OperationConfig::fillParams(b, operands, outParams);
        outParams.immed = mImmediateValue;
    }

  protected:
    uint64_t mImmediateValue;

    using BaseOpConfig::mDescription;
    using BaseOpConfig::mIdentifier;

    bool parseRelevantArgs(cl::list<string> &args, size_t &cur) override {
        unsigned immedVal = 0;
        if (BaseOpConfig::parseRelevantArgs(args, cur) ||
            BaseOpConfig::parseUnsigned(args, cur, "immediate", MinVal, MaxVal, immedVal)) {
            return true;
        }
        mDescription = mDescription + " #" + to_string(immedVal);
        mIdentifier = mIdentifier + "_#" + to_string(immedVal);
        mImmediateValue = immedVal;
        return false;
    }
};

template <class BaseOpConfig, class TestF, class ExpectedF> class GenericOpConfig : public BaseOpConfig {
  public:
    using typename BaseOpConfig::Params;

    GenericOpConfig(StringRef description, StringRef identifier, unsigned fieldWidth, bool quiet, TestF testF,
                    ExpectedF expectedF)
        : BaseOpConfig(description, identifier, fieldWidth), mQuiet(quiet), mTestF(testF), mExpectedF(expectedF) {}

    using BaseOpConfig::fillParams;
    using BaseOpConfig::getDescription;

    Value *makeTestLogic(KernelBuilder &b, vector<llvm::Value *> operands) {
        Params params;
        fillParams(b, operands, params);
        return mTestF(b, *this, params);
    }
    Value *makeCheckLogic(KernelBuilder &b, vector<llvm::Value *> operands) {
        Params params;
        fillParams(b, operands, params);
        return mExpectedF(b, *this, params);
    }

    bool configurePipelineFromArgs(cl::list<string> &args, bool doChecks) override {
        if (BaseOpConfig::configurePipelineFromArgs(args, doChecks)) {
            return true;
        }

        mIdentifier = +(mQuiet ? "_quiet" : "");

        mTestOutput = mPipelineBuilder->CreateStreamSet(1, 1);
        mPipelineBuilder->template CreateKernelCall<TestKernel>(*this, mOperandSSs, mTestOutput);
        if (doChecks) {
            mExpectedOutput = mPipelineBuilder->CreateStreamSet(1, 1);
            mPipelineBuilder->template CreateKernelCall<CheckKernel>(
                *this, mQuiet, mOperandSSs, mTestOutput, mExpectedOutput,
                mPipelineBuilder->getOutputScalar(OperationConfig::failureCountIdent));
        } else {
            // Dummy kernel just to provide a 0 error count
            mPipelineBuilder->template CreateKernelCall<DummyCheckKernel>(
                *this, mTestOutput, mPipelineBuilder->getOutputScalar(OperationConfig::failureCountIdent));
        }
        return false;
    }

  protected:
    using BaseOpConfig::mDescription;
    using BaseOpConfig::mExpectedOutput;
    using BaseOpConfig::mFieldWidth;
    using BaseOpConfig::mIdentifier;
    using BaseOpConfig::mOperandSSs;
    using BaseOpConfig::mPipelineBuilder;
    using BaseOpConfig::mTestOutput;

    bool mQuiet;
    TestF mTestF;
    ExpectedF mExpectedF;
};

// Handy aliases
using UnaryOpConfig = NaryOpConfig<1>;
using BinaryOpConfig = NaryOpConfig<2>;
using TernaryOpConfig = NaryOpConfig<3>;
using ShiftImmedBinOpConfig = ImmediateOpConfig<BinaryOpConfig, 0, 128>;

//////////////////////////////////////////////////////////////////////////////////
// And now, the actual index of operations...

struct OperationIndexEntry {
    typedef bool (*FactoryFunc)(cl::opt<unsigned> &opFieldWidth, cl::opt<bool> &opQuiet,
                                unique_ptr<OperationConfig> &outOpConfig);

    const char *name;
    const char *helpOprs;
    const char *helpDesc;
    FactoryFunc factory;
};

template <size_t N> struct StrucString {
    constexpr StrucString(const char (&str)[N]) { std::copy_n(str, N, value); }

    char value[N];
    auto operator<=>(const StrucString &) const = default;
    bool operator==(const StrucString &) const = default;
};

template <class BaseConfigT, StrucString Name, StrucString HelpOprs, StrucString HelpDesc, auto TestF, auto ExpectedF>
static constexpr OperationIndexEntry genericEntry() {
    return OperationIndexEntry{
        Name.value, HelpOprs.value, HelpDesc.value,
        [](cl::opt<unsigned> &opFieldWidth, cl::opt<bool> &opQuiet, unique_ptr<OperationConfig> &outOpConfig) {
            unsigned fw = opFieldWidth;
            if ((fw < 1) || (fw > 128)) {
                return opFieldWidth.error("Must be in range 1-128");
            }
            if ((fw & (fw - 1)) != 0) {
                return opFieldWidth.error("Must be a power of 2");
            }
            outOpConfig = make_unique<GenericOpConfig<BaseConfigT, decltype(TestF), decltype(ExpectedF)>>(
                Name.value, Name.value, opFieldWidth, opQuiet, TestF, ExpectedF);
            return false;
        }};
}

// I ended up with 2 versions of these wrappers by an accident of refactoring order. It shouldn't be too hard to
// minimize them to 1 shared pair, but it's not quite trivial so dear reader, it's your problem now.

template <class ConfigT, StrucString Name, StrucString HelpOprs, StrucString HelpDesc, auto TestF,
          auto HorizontalStoreExpectedF>
static constexpr OperationIndexEntry horizontalStoreCheckEntry() {
    return genericEntry<ConfigT, Name, HelpOprs, HelpDesc, TestF,
                        [](KernelBuilder &b, const ConfigT &c, const ConfigT::Params &p) -> Value * {
                            unsigned fieldCount = b.getBitBlockWidth() / p.fw;
                            // Since this is for the check, use null to ensure consistent results rather than poison
                            Value *outBlock = Constant::getNullValue(p.vTy);
                            typename ConfigT::Params scalarP = p;
                            for (unsigned i = 0; i < fieldCount; i++) {
                                for (unsigned j = 0; j < size(scalarP.opr); ++j) {
                                    scalarP.opr[j] = b.mvmd_extract(p.fw, p.opr[j], i);
                                }
                                outBlock = HorizontalStoreExpectedF(b, c, scalarP, outBlock, i);
                            }
                            return outBlock;
                        }>();
}

template <class ConfigT, StrucString Name, StrucString HelpOprs, StrucString HelpDesc, auto TestF, auto ScalarExpectedF>
static constexpr OperationIndexEntry scalarCheckEntry() {
    return horizontalStoreCheckEntry<ConfigT, Name, HelpOprs, HelpDesc, TestF,
                                     [](KernelBuilder &b, const ConfigT &c, const ConfigT::Params &p, Value *outBlock,
                                        unsigned i) -> Value * {
                                         Value *out = ScalarExpectedF(b, c, p);
                                         return b.bitCast(b.mvmd_insert(p.fw, outBlock, out, i));
                                     }>();
}

// Sort of redundant with the above, oops

template <class ConfigT, class HorizontalStoreCheckF>
auto wrapHorizontalStore(HorizontalStoreCheckF &&horizontalStoreCheckF) {
    return [=](KernelBuilder &b, const ConfigT &c, const ConfigT::Params &p) {
        unsigned fieldCount = b.getBitBlockWidth() / p.fw;
        Value *expectedBlock = Constant::getNullValue(b.fwVectorType(p.fw));
        for (unsigned i = 0; i < fieldCount; i++) {
            typename ConfigT::Params scalarP = p;
            for (unsigned j = 0; j < size(scalarP.opr); ++j) {
                scalarP.opr[j] = b.mvmd_extract(p.fw, p.opr[j], i);
            }

            expectedBlock = horizontalStoreCheckF(b, c, scalarP, expectedBlock, i);
        }
        return expectedBlock;
    };
};

template <class ConfigT, class ScalarCheckF> auto wrapScalar(ScalarCheckF &&scalarCheckF) {
    return wrapHorizontalStore<ConfigT>(
        [=](KernelBuilder &b, const ConfigT &c, const ConfigT::Params &p, Value *expectedBlock, unsigned i) {
            Value *expected = scalarCheckF(b, c, p);
            return b.bitCast(b.mvmd_insert(p.fw, expectedBlock, expected, i));
        });
}

OperationIndexEntry allOperations[] = {
    scalarCheckEntry<BinaryOpConfig, "simd_add", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_add(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateAdd(p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_sub", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_sub(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSub(p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_mult", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_mult(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateMul(p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_eq", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_eq(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpEQ(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_ne", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_ne(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpNE(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_gt", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_gt(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpSGT(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_ugt", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_ugt(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpUGT(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_ge", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_ge(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpSGE(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_uge", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_uge(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpUGE(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_lt", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_lt(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpSLT(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_le", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_le(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpSLE(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_ult", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_ult(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpULT(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_ule", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_ule(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSExt(b.CreateICmpULE(p.opr[0], p.opr[1]), p.fTy);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_max", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_max(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSelect(b.CreateICmpSGT(p.opr[0], p.opr[1]), p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_min", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_min(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSelect(b.CreateICmpSLT(p.opr[0], p.opr[1]), p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_umax", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_umax(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSelect(b.CreateICmpUGT(p.opr[0], p.opr[1]), p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_umin", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_umin(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateSelect(b.CreateICmpULT(p.opr[0], p.opr[1]), p.opr[0], p.opr[1]);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_sllv", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_sllv(p.fw, p.opr[0],
                                            b.CreateAnd(b.fwCast(p.fw, p.opr[1]), b.getSplatN(p.fw, p.fn, p.fw - 1)));
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateShl(p.opr[0], b.CreateAnd(p.opr[1], b.getIntN(p.fw, p.fw - 1)));
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_srlv", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_srlv(p.fw, p.opr[0],
                                            b.CreateAnd(b.fwCast(p.fw, p.opr[1]), b.getSplatN(p.fw, p.fn, p.fw - 1)));
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.CreateLShr(p.opr[0], b.CreateAnd(p.opr[1], b.getIntN(p.fw, p.fw - 1)));
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_rotl", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_rotl(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         Constant *fwConst = ConstantInt::get(p.fTy, p.fw);
                         Constant *fwMaskConst = ConstantInt::get(p.fTy, p.fw - 1);
                         Value *shl = b.CreateShl(p.opr[0], b.CreateAnd(p.opr[1], fwMaskConst));
                         Value *shr = b.CreateLShr(p.opr[0], b.CreateAnd(b.CreateSub(fwConst, p.opr[1]), fwMaskConst));
                         return b.CreateOr(shl, shr);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_rotr", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_rotr(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         Constant *fwConst = b.getIntN(p.fw, p.fw);
                         Constant *fwMaskConst = b.getIntN(p.fw, p.fw - 1);
                         Value *shl = b.CreateShl(p.opr[0], b.CreateAnd(b.CreateSub(fwConst, p.opr[1]), fwMaskConst));
                         Value *shr = b.CreateLShr(p.opr[0], b.CreateAnd(p.opr[1], fwMaskConst));
                         return b.CreateOr(shl, shr);
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_pext", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_pext(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         Constant *zeroConst = b.getIntN(p.fw, 0);
                         Constant *oneConst = b.getIntN(p.fw, 1);
                         Value *expected = zeroConst;
                         Value *out_bit = oneConst;
                         for (unsigned i = 0; i < p.fw; i++) {
                             Value *i_bit = b.getIntN(p.fw, 1LL << i);
                             Value *operand_i_isSet = b.CreateICmpEQ(b.CreateAnd(p.opr[0], i_bit), i_bit);
                             Value *mask_i_isSet = b.CreateICmpEQ(b.CreateAnd(p.opr[1], i_bit), i_bit);
                             expected = b.CreateSelect(b.CreateAnd(operand_i_isSet, mask_i_isSet),
                                                       b.CreateOr(expected, out_bit), expected);
                             out_bit = b.CreateSelect(mask_i_isSet, b.CreateAdd(out_bit, out_bit), out_bit);
                         }
                         return expected;
                     }>(),
    scalarCheckEntry<BinaryOpConfig, "simd_pdep", "x0, x1", "",
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         return b.simd_pdep(p.fw, p.opr[0], p.opr[1]);
                     },
                     [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                         Constant *zeroConst = ConstantInt::getNullValue(p.fTy);
                         Constant *oneConst = ConstantInt::get(p.fTy, 1);
                         Value *expected = zeroConst;
                         Value *shft = zeroConst;
                         Value *select_bit = oneConst;
                         for (unsigned i = 0; i < p.fw; i++) {
                             expected = b.CreateOr(
                                 b.CreateAnd(p.opr[1], b.CreateShl(b.CreateAnd(p.opr[0], select_bit), shft)), expected);
                             Value *i_bit = b.getIntN(p.fw, 1LL << i);
                             Value *mask_i_isSet = b.CreateICmpEQ(b.CreateAnd(p.opr[1], i_bit), i_bit);
                             select_bit = b.CreateSelect(mask_i_isSet, b.CreateAdd(select_bit, select_bit), select_bit);
                             shft = b.CreateSelect(mask_i_isSet, shft, b.CreateAdd(shft, oneConst));
                         }
                         return expected;
                     }>(),
    horizontalStoreCheckEntry<BinaryOpConfig, "hsimd_packh", "x0, x1", "",
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                                  return b.hsimd_packh(p.fw, p.opr[0], p.opr[1]);
                              },
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p,
                                 Value *expectedBlock, unsigned i) {
                                  Value *newOpr0 =
                                      b.CreateTrunc(b.CreateLShr(p.opr[0], p.fw / 2), b.getIntNTy(p.fw / 2));
                                  Value *newOpr1 =
                                      b.CreateTrunc(b.CreateLShr(p.opr[1], p.fw / 2), b.getIntNTy(p.fw / 2));
                                  expectedBlock = b.mvmd_insert(p.fw / 2, expectedBlock, newOpr0, i);
                                  expectedBlock = b.bitCast(b.mvmd_insert(p.fw / 2, expectedBlock, newOpr1, p.fn + i));
                                  return expectedBlock;
                              }>(),
    horizontalStoreCheckEntry<BinaryOpConfig, "hsimd_packl", "x0, x1", "",
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                                  return b.hsimd_packl(p.fw, p.opr[0], p.opr[1]);
                              },
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p,
                                 Value *expectedBlock, unsigned i) {
                                  Value *newOpr0 = b.CreateTrunc(p.opr[0], b.getIntNTy(p.fw / 2));
                                  Value *newOpr1 = b.CreateTrunc(p.opr[1], b.getIntNTy(p.fw / 2));
                                  expectedBlock = b.mvmd_insert(p.fw / 2, expectedBlock, newOpr0, i);
                                  expectedBlock = b.bitCast(b.mvmd_insert(p.fw / 2, expectedBlock, newOpr1, p.fn + i));
                                  return expectedBlock;
                              }>(),
    genericEntry<BinaryOpConfig, "hsimd_packus", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     Value *newOpr0 = b.simd_srai(p.fw, p.opr[0], p.fw / 2 - 1);
                     Value *newOpr1 = b.simd_srai(p.fw, p.opr[1], p.fw / 2 - 1);
                     return b.hsimd_packus(p.fw, newOpr0, newOpr1);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     BinaryOpConfig::Params newP = p;
                     newP.opr[0] = b.simd_srai(p.fw, p.opr[0], p.fw / 2 - 1);
                     newP.opr[1] = b.simd_srai(p.fw, p.opr[1], p.fw / 2 - 1);
                     auto expectedF = wrapHorizontalStore<BinaryOpConfig>([=](KernelBuilder &b, const BinaryOpConfig &c,
                                                                              const BinaryOpConfig::Params &p,
                                                                              Value *expectedBlock, unsigned i) {
                         Value *zeroes = ConstantInt::getNullValue(p.fTy);
                         Value *newOpr0 = b.CreateSelect(b.CreateICmpSLT(p.opr[0], zeroes), zeroes, p.opr[0]);
                         Value *newOpr1 = b.CreateSelect(b.CreateICmpSLT(p.opr[1], zeroes), zeroes, p.opr[1]);
                         Value *testVal = ConstantInt::get(b.getContext(), APInt::getLowBitsSet(p.fw, p.fw / 2));
                         newOpr0 = b.CreateSelect(b.CreateICmpSGT(newOpr0, testVal), testVal, newOpr0);
                         newOpr1 = b.CreateSelect(b.CreateICmpSGT(newOpr1, testVal), testVal, newOpr1);
                         newOpr0 = b.CreateTrunc(newOpr0, b.getIntNTy(p.fw / 2));
                         newOpr1 = b.CreateTrunc(newOpr1, b.getIntNTy(p.fw / 2));
                         expectedBlock = b.mvmd_insert(p.fw / 2, expectedBlock, newOpr0, i);
                         expectedBlock = b.bitCast(b.mvmd_insert(p.fw / 2, expectedBlock, newOpr1, p.fn + i));
                         return expectedBlock;
                     });
                     return expectedF(b, c, newP);
                 }>(),
    genericEntry<BinaryOpConfig, "hsimd_packss", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     Value *newOpr0 = b.simd_srai(p.fw, p.opr[0], p.fw / 2 - 1);
                     Value *newOpr1 = b.simd_srai(p.fw, p.opr[1], p.fw / 2 - 1);
                     return b.hsimd_packss(p.fw, newOpr0, newOpr1);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     BinaryOpConfig::Params newP = p;
                     newP.opr[0] = b.simd_srai(p.fw, p.opr[0], p.fw / 2 - 1);
                     newP.opr[1] = b.simd_srai(p.fw, p.opr[1], p.fw / 2 - 1);
                     auto expectedF = wrapHorizontalStore<BinaryOpConfig>([=](KernelBuilder &b, const BinaryOpConfig &c,
                                                                              const BinaryOpConfig::Params &p,
                                                                              Value *expectedBlock, unsigned i) {
// JL -- I found 2 versions of the hsimd_packss test, the first is the one that would have been active...?
#if 1
                         Value *maxVal = ConstantInt::get(b.getContext(), APInt::getLowBitsSet(p.fw, p.fw / 2 - 1));
                         Value *newOpr0 = b.CreateSelect(b.CreateICmpSGT(p.opr[0], maxVal), maxVal, p.opr[0]);
                         Value *newOpr1 = b.CreateSelect(b.CreateICmpSGT(p.opr[1], maxVal), maxVal, p.opr[1]);
                         Value *minVal = ConstantInt::get(b.getContext(), APInt::getHighBitsSet(p.fw, p.fw / 2 + 1));
                         newOpr0 = b.CreateSelect(b.CreateICmpSLT(newOpr0, minVal), minVal, p.opr[0]);
                         newOpr1 = b.CreateSelect(b.CreateICmpSLT(newOpr1, minVal), minVal, newOpr1);
                         newOpr0 = b.CreateTrunc(newOpr0, b.getIntNTy(p.fw / 2));
                         newOpr1 = b.CreateTrunc(newOpr1, b.getIntNTy(p.fw / 2));
                         expectedBlock = b.mvmd_insert(p.fw / 2, expectedBlock, newOpr0, i);
                         expectedBlock = b.bitCast(b.mvmd_insert(p.fw / 2, expectedBlock, newOpr1, p.fn + i));
#else
                         Value *testVal = ConstantInt::get(p.fTy, (1 << (p.fw / 2 - 1)) - 1);
                         Value *newOpr0 = b.CreateSelect(b.CreateICmpSGT(p.opr[0], testVal), testVal, p.opr[0]);
                         Value *newOpr1 = b.CreateSelect(b.CreateICmpSGT(p.opr[1], testVal), testVal, p.opr[1]);
                         testVal = b.CreateNot(testVal);
                         newOpr0 = b.CreateSelect(b.CreateICmpSLT(newOpr0, testVal), testVal, newOpr0);
                         newOpr1 = b.CreateSelect(b.CreateICmpSLT(newOpr1, testVal), testVal, newOpr1);
                         expectedBlock = b.mvmd_insert(p.fw / 2, expectedBlock, newOpr0, i);
                         expectedBlock = b.bitCast(b.mvmd_insert(p.fw / 2, expectedBlock, newOpr1, p.fn + i));
#endif
                         return expectedBlock;
                     });
                     return expectedF(b, c, newP);
                 }>(),
    horizontalStoreCheckEntry<BinaryOpConfig, "esimd_mergeh", "x0, x1", "",
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                                  return b.esimd_mergeh(p.fw, p.opr[0], p.opr[1]);
                              },
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p,
                                 Value *expectedBlock, unsigned i) {
                                  if (i >= p.fn / 2) {
                                      expectedBlock = b.mvmd_insert(p.fw, expectedBlock, p.opr[0], 2 * (i - p.fn / 2));
                                      expectedBlock = b.bitCast(
                                          b.mvmd_insert(p.fw, expectedBlock, p.opr[1], 2 * (i - p.fn / 2) + 1));
                                  }
                                  return expectedBlock;
                              }>(),
    horizontalStoreCheckEntry<BinaryOpConfig, "esimd_mergel", "x0, x1", "",
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                                  return b.esimd_mergel(p.fw, p.opr[0], p.opr[1]);
                              },
                              [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p,
                                 Value *expectedBlock, unsigned i) {
                                  if (i < p.fn / 2) {
                                      expectedBlock = b.mvmd_insert(p.fw, expectedBlock, p.opr[0], 2 * i);
                                      expectedBlock =
                                          b.bitCast(b.mvmd_insert(p.fw, expectedBlock, p.opr[1], 2 * i + 1));
                                  }
                                  return expectedBlock;
                              }>(),
    genericEntry<BinaryOpConfig, "mvmd_shuffle", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     return b.mvmd_shuffle(p.fw, p.opr[0], p.opr[1], IDISA::ShuffleMode::TruncateIndex);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     Value *expectedBlock = Constant::getNullValue(p.vTy);
                     for (unsigned i = 0; i < p.fn; i++) {
                         Value *idx_field = b.mvmd_extract(p.fw, p.opr[1], i);
                         Value *idx = b.CreateURem(idx_field, ConstantInt::get(p.fTy, p.fn));
                         Value *elt =
                             b.CreateExtractElement(b.fwCast(p.fw, p.opr[0]), b.CreateZExtOrTrunc(idx, p.i32Ty));
                         // ShuffleMode::TruncateIndex behaviour
                         expectedBlock = b.mvmd_insert(p.fw, expectedBlock, elt, i);
                     }
                     return expectedBlock;
                 }>(),
    genericEntry<BinaryOpConfig, "mvmd_shuffle:over", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     return b.mvmd_shuffle(p.fw, p.opr[0], p.opr[1], IDISA::ShuffleMode::ZeroOnIndexOver);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     Constant *fieldLimit = ConstantInt::get(p.fTy, p.fn);
                     Value *expectedBlock = Constant::getNullValue(p.vTy);
                     for (unsigned i = 0; i < p.fn; i++) {
                         Value *idx_field = b.mvmd_extract(p.fw, p.opr[1], i);
                         Value *idx = b.CreateURem(idx_field, ConstantInt::get(p.fTy, p.fn));
                         Value *elt =
                             b.CreateExtractElement(b.fwCast(p.fw, p.opr[0]), b.CreateZExtOrTrunc(idx, p.i32Ty));
                         // ShuffleMode::ZeroOnIndexOver behaviour
                         elt = b.CreateSelect(b.CreateICmpUGE(idx_field, fieldLimit), ConstantInt::getNullValue(p.fTy),
                                              elt);
                         expectedBlock = b.mvmd_insert(p.fw, expectedBlock, elt, i);
                     }
                     return expectedBlock;
                 }>(),
    genericEntry<BinaryOpConfig, "mvmd_shuffle:highbit", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     return b.mvmd_shuffle(p.fw, p.opr[0], p.opr[1], IDISA::ShuffleMode::ZeroOnHighIndexBit);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     Value *expectedBlock = Constant::getNullValue(p.vTy);
                     for (unsigned i = 0; i < p.fn; i++) {
                         Value *idx_field = b.mvmd_extract(p.fw, p.opr[1], i);
                         Value *idx = b.CreateURem(idx_field, ConstantInt::get(p.fTy, p.fn));
                         Value *elt =
                             b.CreateExtractElement(b.fwCast(p.fw, p.opr[0]), b.CreateZExtOrTrunc(idx, p.i32Ty));
                         // ShuffleMode::ZeroOnHighIndexBit behaviour
                         elt = b.CreateSelect(b.CreateICmpSLT(idx_field, ConstantInt::getNullValue(p.fTy)),
                                              ConstantInt::getNullValue(p.fTy), elt);
                         expectedBlock = b.mvmd_insert(p.fw, expectedBlock, elt, i);
                     }
                     return expectedBlock;
                 }>(),
    genericEntry<BinaryOpConfig, "mvmd_compress", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     // Real callers (e.g. deletion.cpp) pass a scalar bitmask built with
                     // hsimd_signmask, not a raw data block. Derive a realistic one from
                     // p.opr[1] here so this test actually matches production usage.
                     Value *scalarMask = b.hsimd_signmask(p.fw, p.opr[1]);
                     return b.mvmd_compress(p.fw, p.opr[0], scalarMask);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     // Match the scalar bitmask built in IdisaBinaryOpTestKernel above.
                     Value *scalarMask = b.hsimd_signmask(p.fw, p.opr[1]);
                     Type *maskTy = scalarMask->getType();
                     // For each input field i, precompute whether it's selected and its
                     // rank (how many selected fields come before it) - this is the
                     // output slot it should land in if selected.
                     std::vector<Value *> isSelected(p.fn);
                     std::vector<Value *> rank(p.fn);
                     Value *runningCount = ConstantInt::get(p.i32Ty, 0);
                     Value *expectedBlock = Constant::getNullValue(p.vTy);
                     for (unsigned i = 0; i < p.fn; i++) {
                         Value *bit = b.CreateAnd(b.CreateLShr(scalarMask, ConstantInt::get(maskTy, i)),
                                                  ConstantInt::get(maskTy, 1));
                         isSelected[i] = b.CreateICmpNE(bit, ConstantInt::get(maskTy, 0));
                         rank[i] = runningCount;
                         runningCount = b.CreateAdd(runningCount, b.CreateZExt(isSelected[i], p.i32Ty));
                     }
                     // For each output slot j, find the (at most one) selected input
                     // field whose rank equals j, and place its value there.
                     for (unsigned j = 0; j < p.fn; j++) {
                         Value *chosen = ConstantInt::get(p.fTy, 0);
                         for (unsigned i = 0; i < p.fn; i++) {
                             Value *matches =
                                 b.CreateAnd(isSelected[i], b.CreateICmpEQ(rank[i], ConstantInt::get(p.i32Ty, j)));
                             Value *elt = b.mvmd_extract(p.fw, p.opr[0], i);
                             chosen = b.CreateSelect(matches, elt, chosen);
                         }
                         expectedBlock = b.mvmd_insert(p.fw, expectedBlock, chosen, j);
                     }
                     return expectedBlock;
                 }>(),
    genericEntry<BinaryOpConfig, "mvmd_expand", "x0, x1", "",
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     Value *scalarMask = b.hsimd_signmask(p.fw, p.opr[1]);
                     return b.mvmd_expand(p.fw, p.opr[0], scalarMask);
                 },
                 [](KernelBuilder &b, const BinaryOpConfig &c, const BinaryOpConfig::Params &p) {
                     // Mirror image of mvmd_compress's reference above: for each output
                     // slot j, if it's selected, it should hold p.opr[0]'s field at
                     // rank(j) (how many selected slots come before j); otherwise 0.
                     Value *scalarMask = b.hsimd_signmask(p.fw, p.opr[1]);
                     Type *maskTy = scalarMask->getType();
                     std::vector<Value *> isSelected(p.fn);
                     std::vector<Value *> rank(p.fn);
                     Value *runningCount = ConstantInt::get(p.i32Ty, 0);
                     Value *expectedBlock = Constant::getNullValue(p.vTy);
                     for (unsigned j = 0; j < p.fn; j++) {
                         Value *bit = b.CreateAnd(b.CreateLShr(scalarMask, ConstantInt::get(maskTy, j)),
                                                  ConstantInt::get(maskTy, 1));
                         isSelected[j] = b.CreateICmpNE(bit, ConstantInt::get(maskTy, 0));
                         rank[j] = runningCount;
                         runningCount = b.CreateAdd(runningCount, b.CreateZExt(isSelected[j], p.i32Ty));
                     }
                     for (unsigned j = 0; j < p.fn; j++) {
                         Value *chosen = ConstantInt::get(p.fTy, 0);
                         for (unsigned k = 0; k < p.fn; k++) {
                             Value *matches =
                                 b.CreateAnd(isSelected[j], b.CreateICmpEQ(rank[j], ConstantInt::get(p.i32Ty, k)));
                             Value *elt = b.mvmd_extract(p.fw, p.opr[0], k);
                             chosen = b.CreateSelect(matches, elt, chosen);
                         }
                         expectedBlock = b.mvmd_insert(p.fw, expectedBlock, chosen, j);
                     }
                     return expectedBlock;
                 }>(),
    genericEntry<ShiftImmedBinOpConfig, "mvmd_dslli", "x0, x1, imm", "",
                 [](KernelBuilder &b, const ShiftImmedBinOpConfig &c, const ShiftImmedBinOpConfig::Params &p) {
                     return b.mvmd_dslli(p.fw, p.opr[0], p.opr[1], p.immed);
                 },
                 [](KernelBuilder &b, const ShiftImmedBinOpConfig &c, const ShiftImmedBinOpConfig::Params &p) {
                     Value *expectedBlock = Constant::getNullValue(p.vTy);
                     for (unsigned i = 0; i < p.fn; i++) {
                         Value *elt = nullptr;
                         if (i < p.immed)
                             elt = b.mvmd_extract(p.fw, p.opr[1], p.fn - p.immed + i);
                         else
                             elt = b.mvmd_extract(p.fw, p.opr[0], i - p.immed);
                         expectedBlock = b.mvmd_insert(p.fw, expectedBlock, elt, i);
                     }
                     return expectedBlock;
                 }>(),
};

/*
    llvm::Value *simd_fill(unsigned fw, llvm::Value *a)

    llvm::Value *simd_if(unsigned fw, llvm::Value *cond, llvm::Value *a, llvm::Value *b)
    llvm::Value *simd_binary(unsigned char mask, llvm::Value *bit_1, llvm::Value *bit_0)
    llvm::Value *simd_ternary(unsigned char mask, llvm::Value *bit_2, llvm::Value *bit_1, llvm::Value *bit_0)

    llvm::Value *simd_slli(unsigned fw, llvm::Value *a, unsigned shift)
    llvm::Value *simd_srli(unsigned fw, llvm::Value *a, unsigned shift)

    llvm::Value *simd_any(unsigned fw, llvm::Value *a)
    llvm::Value *simd_popcount(unsigned fw, llvm::Value *a)
    llvm::Value *hsimd_partial_sum(unsigned fw, llvm::Value *a)
    llvm::Value *simd_cttz(unsigned fw, llvm::Value *a)

    llvm::Value *simd_bitreverse(unsigned fw, llvm::Value *a)

    llvm::Value *hsimd_packh_in_lanes(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b)
    llvm::Value *hsimd_packl_in_lanes(unsigned lanes, unsigned fw, llvm::Value *a, llvm::Value *b)

    llvm::Value *hsimd_signmask(unsigned fw, llvm::Value *a)

    llvm::Value *mvmd_extract(unsigned fw, llvm::Value *a, unsigned fieldIndex)
    llvm::Value *mvmd_insert(unsigned fw, llvm::Value *blk, llvm::Value *elt, unsigned fieldIndex)

    llvm::Value *mvmd_sll(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe = false)
    llvm::Value *mvmd_srl(unsigned fw, llvm::Value *value, llvm::Value *shift, const bool safe = false)
    llvm::Value *mvmd_slli(unsigned fw, llvm::Value *a, unsigned shift)
    llvm::Value *mvmd_srli(unsigned fw, llvm::Value *a, unsigned shift)
    //llvm::Value *mvmd_dslli(unsigned fw, llvm::Value *a, llvm::Value *b, unsigned shift)
    llvm::Value *mvmd_dsll(unsigned fw, llvm::Value *a, llvm::Value *b, llvm::Value *shift)

    llvm::Value *mvmd_shuffle2(unsigned fw, llvm::Value *table0, llvm::Value *table1, llvm::Value *index_vector,
                               ShuffleMode m = ShuffleMode::TruncateIndex)

    llvm::Value *bitblock_any(llvm::Value *a)
    std::pair<llvm::Value *, llvm::Value *> bitblock_add_with_carry(llvm::Value *a, llvm::Value *b,
                                                                    llvm::Value *carryin)
    std::pair<llvm::Value *, llvm::Value *> bitblock_subtract_with_borrow(llvm::Value *a, llvm::Value *b,
                                                                          llvm::Value *borrowin)
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance(llvm::Value *a, llvm::Value *shiftin, unsigned shift)
    std::pair<llvm::Value *, llvm::Value *> bitblock_indexed_advance(llvm::Value *a, llvm::Value *index_strm,
                                                                     llvm::Value *shiftin, unsigned shift)

    llvm::Value *bitblock_mask_from(llvm::Value *const position, const bool safe = false)
    llvm::Value *bitblock_mask_to(llvm::Value *const position, const bool safe = false)
    llvm::Value *bitblock_set_bit(llvm::Value *const position, const bool safe = false)

    llvm::Value *bitblock_popcount(llvm::Value *const to_count)

    llvm::Value *simd_not(llvm::Value *a, llvm::StringRef s = llvm::StringRef());
*/

// static cl::opt<IDISA::ShuffleMode>
//     ShuffleIndex("ShuffleIndex",
//                  cl::values(clEnumValN(IDISA::ShuffleMode::TruncateIndex, "Truncate",
//                                        "Truncate out-of-bound shuffle indexes."),
//                             clEnumValN(IDISA::ShuffleMode::ZeroOnIndexOver, "ZeroOnOver",
//                                        "Select zero for shuffle indexes out of bound."),
//                             clEnumValN(IDISA::ShuffleMode::ZeroOnHighIndexBit, "ZeroOnHighBit",
//                                        "Select zero if high index bit set, otherwise truncate.")),
//                  cl::init(IDISA::IDISA_Builder::ShuffleMode::TruncateIndex));

vector<tuple<StringRef, StringRef, StringRef>> allOperationHelpDescs() {
    vector<tuple<StringRef, StringRef, StringRef>> allOps;
    for (auto const &entry : allOperations) {
        allOps.emplace_back(entry.name, entry.helpOprs, entry.helpDesc);
    }
    return allOps;
}

bool makeOperationConfig(cl::opt<string> &opName, cl::opt<unsigned> &opFieldWidth, cl::opt<bool> &opQuiet,
                         unique_ptr<OperationConfig> &outOpConfig) {
    for (auto const &entry : allOperations) {
        if (opName == entry.name) {
            return entry.factory(opFieldWidth, opQuiet, outOpConfig);
        }
    }
    return opName.error("Operation " + opName + " unknown or not implemented yet");
}
