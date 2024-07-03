#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/relationship.h>
#include <kernel/io/source_kernel.h>

using namespace kernel;
using namespace llvm;

namespace audio
{
    void readWAVHeader(
        const int &fd,
        unsigned int &numChannels,
        unsigned int &sampleRate,
        unsigned int &bitPerSample,
        unsigned int &numSamples);

    void ExtractWAVData(
        const std::unique_ptr<ProgramBuilder> &P,
        Scalar *const fileDescriptor,
        unsigned int numChannels,
        unsigned int numSamples,
        unsigned int sampleRate,
        unsigned int bitPerSample,
        const bool includedHeader,
        StreamSet *&outputDataStreams);

    class Stereo2MonoKernel final : public MultiBlockKernel {
    public:
        Stereo2MonoKernel(KernelBuilder & b,
                StreamSet * const inputStreams,
                StreamSet * const outputStream,
                const unsigned int bitsPerSample = 16);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
    };

    class AmplifyKernel final : public MultiBlockKernel {
    public:
        AmplifyKernel(KernelBuilder & b,
                StreamSet * const inputStreams,
                const unsigned int& factor,
                StreamSet * const outputStreams,
                const unsigned int bitsPerSample = 16);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
        unsigned int factor;
    };
}
