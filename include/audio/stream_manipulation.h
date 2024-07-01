#pragma once
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/relationship.h>
#include <llvm/IR/Value.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>

using namespace pablo;
using namespace kernel;
using namespace llvm;

namespace audio 
{
    class CreateOnes : public PabloKernel {
    public:
        CreateOnes(KernelBuilder & kb, StreamSet * dataStream, StreamSet * onesStream)
            : PabloKernel(kb, "CreateOnes",
                        {Binding{"dataStream", dataStream}},
                        {Binding{"onesStream", onesStream}}) {}
    protected:
        void generatePabloMethod() override;
    };

    class mS2PKernel final : public MultiBlockKernel {
    public:
        mS2PKernel(KernelBuilder & b,
                StreamSet * const inputStreams,
                StreamSet * const outputStreams,
                const unsigned int bitsPerSample = 16);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
    };
}