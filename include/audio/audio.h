#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/relationship.h>
#include <kernel/io/source_kernel.h>
#include <pablo/builder.hpp>

using namespace kernel;
using namespace llvm;
using namespace pablo;

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

    void S2P(
        const std::unique_ptr<ProgramBuilder> &P,
        unsigned int bitPerSample,
        StreamSet * const inputStream,
        StreamSet *&outputStreams);

    void P2S(
        const std::unique_ptr<ProgramBuilder> &P,
        StreamSet * const inputStreams,
        StreamSet *&outputStream);

    class FlexS2PKernel final : public MultiBlockKernel {
    public:
        FlexS2PKernel(kernel::KernelBuilder & b, 
                const unsigned int bitsPerSample,
                StreamSet * const inputStream,
                StreamSet * const outputStreams);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
    };

    class Stereo2MonoKernel final : public MultiBlockKernel {
    public:
        Stereo2MonoKernel(kernel::KernelBuilder & b,
                const unsigned int bitsPerSample,
                StreamSet * const inputStreams,
                StreamSet * const outputStream);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
    };

    class Stereo2MonoPabloKernel final : public PabloKernel {
    public:
        Stereo2MonoPabloKernel(kernel::KernelBuilder & b,
                StreamSet * const firstInputStreams,
                StreamSet * const secondInputStreams,
                StreamSet * const outputStreams);
    protected:
        void generatePabloMethod() override;
    };


    class AmplifyKernel final : public MultiBlockKernel {
    public:
        AmplifyKernel(kernel::KernelBuilder & b,
                const unsigned int bitsPerSample,
                StreamSet * const inputStreams,
                const unsigned int& factor,
                StreamSet * const outputStreams);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
        unsigned int factor;
    };

    class AmplifyPabloKernel final : public PabloKernel {
    public:
        AmplifyPabloKernel(kernel::KernelBuilder & b,
                const unsigned int bitsPerSample,
                StreamSet * const inputStreams,
                const unsigned int& factor,
                StreamSet * const outputStreams);
    protected:
        void generatePabloMethod() override;

    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
        unsigned int factor;
    };

    class ConcatenateKernel final : public PabloKernel {
    public:
        ConcatenateKernel(kernel::KernelBuilder & b,
                StreamSet *const firstInputStreams,
                StreamSet *const secondInputStreams,
                StreamSet * const outputStreams);
    protected:
        void generatePabloMethod() override;

    private:
        unsigned int numFirstInputStreams;
        unsigned int numSecondInputStreams;
    };
}
