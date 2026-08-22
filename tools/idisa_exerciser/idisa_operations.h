#pragma once

#include "idisa_exerciser.h"

namespace llvm {
class Type;
class Value;
} // namespace llvm

class OperationConfig {
  public:
    struct Params {
        unsigned fw;       // Field width
        unsigned fn;       // Field count (num)
        llvm::Type *fTy;   // Field type (fw-width int)
        llvm::Type *vTy;   // Vector type (fn x fw)
        llvm::Type *i32Ty; // Just handy: int32 type
    };

    static std::string operandFDIdent(unsigned index) { return std::format("operand{}_fd", index + 1); }
    static std::string operandIdent(unsigned index) { return std::format("operand{}", index + 1); }
    static constexpr const char *testOutputIdent = "test_output";
    static constexpr const char *expectedOutputIdent = "expected_output";
    static constexpr const char *failureCountIdent = "failure_count";
    static constexpr const char *outputFilenameIdent = "output_filename";

    explicit OperationConfig(llvm::StringRef description, llvm::StringRef identifier, unsigned fieldWidth)
        : mDescription(description.str() + " x i" + std::to_string(fieldWidth)),
          mIdentifier(identifier.str() + "_" + std::to_string(fieldWidth)), mFieldWidth(fieldWidth) {}
    virtual ~OperationConfig();

    // String describing how the operation is configured, for logging etc.
    llvm::StringRef getDescription() const { return mDescription; }

    // A more compact name meant to use for e.g. the kernel name
    llvm::StringRef getIdentifier() const { return mIdentifier; }

    unsigned getFieldWidth() const { return mFieldWidth; }

    virtual void constructPipeline(CPUDriver &driver) = 0;
    virtual bool configurePipelineFromArgs(llvm::cl::list<std::string> &args);
    virtual void compilePipeline() = 0;
    virtual size_t executePipeline() = 0;

    kernel::PipelineBuilder &getPipelineBuilder() {
        assert(mPipelineBuilder);
        return *mPipelineBuilder;
    }
    kernel::StreamSet *getOutputStream() const { return mTestOutput; }
    void setOutputFileName(llvm::StringRef outputFilename) {
        assert(!mExecuteFunc);
        mOutputFilename = outputFilename.str();
    }

    bool isStdinGrabbed() const { return mStdinGrabbed; }

    virtual llvm::Value *makeTestLogic(kernel::KernelBuilder &b, std::vector<llvm::Value *> operands) = 0;
    virtual llvm::Value *makeCheckLogic(kernel::KernelBuilder &b, std::vector<llvm::Value *> operands) = 0;

    void fillParams(kernel::KernelBuilder &b, std::vector<llvm::Value *> operands, Params &outParams) const;

  protected:
    std::string mDescription;
    std::string mIdentifier;

    unsigned mFieldWidth;
    std::vector<kernel::StreamSet *> mOperandSSs;
    std::string mOutputFilename;
    kernel::StreamSet *mTestOutput;
    kernel::StreamSet *mExpectedOutput;

    bool mStdinGrabbed = false;

    std::unique_ptr<kernel::PipelineBuilder> mPipelineBuilder;
#ifndef NDEBUG
    // Make it obvious that if asserts are disabled we won't check this anyway
    bool mPipelineConfigured = false;
#endif
    std::function<size_t()> mExecuteFunc;
    std::vector<std::function<void()>> mResetFuncs;
    std::vector<std::function<void()>> mDtorFuncs;

    // Parse args that are relevant to this particular type: we might be called as a parent type, so if there are some
    // left over we leave it to the child to decide what to do with them. If they're genuinely extra,
    // configurePipelineFromArgs() (who will have indirectly called us) will notice
    virtual bool parseRelevantArgs(llvm::cl::list<std::string> &args, size_t &cur);
    bool parseUnsigned(llvm::cl::list<std::string> &args, size_t &cur, llvm::StringRef expectName, unsigned minVal,
                       unsigned maxVal, unsigned &outResult);
    bool parseStreamSource(llvm::cl::list<std::string> &args, size_t &cur, llvm::StringRef expectName,
                           llvm::StringRef fdIdent, int32_t &outFDResult, kernel::StreamSet *&outSSResult);
    bool openInputFile(llvm::cl::Option &arg, llvm::StringRef fileName, int32_t &outFDResult);

    // Child should before executing in executePipeline()
    virtual void resetPipeline();

    // Noncopyable
    OperationConfig(const OperationConfig &) = delete;
    OperationConfig &operator=(const OperationConfig &) = delete;
};

std::vector<std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>> allOperationHelpDescs();
bool makeOperationConfig(llvm::cl::opt<std::string> &opName, llvm::cl::opt<unsigned> &opFieldWidth,
                         llvm::cl::opt<bool> &opQuiet, std::unique_ptr<OperationConfig> &outOpConfig);
