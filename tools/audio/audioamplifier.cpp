#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/streamutils/stream_select.h>
#include <string>
#include <toolchain/toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <audio/audio.h>
#include <audio/stream_manipulation.h>
#include <iostream>
#include <util/aligned_allocator.h>

using namespace kernel;
using namespace llvm;
using namespace codegen;
using namespace audio;

#define SHOW_STREAM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBixNum(#name, name)
#define SHOW_BYTES(name)            \
    if (codegen::EnableIllustrator) \
    P->captureByteData(#name, name)

static cl::OptionCategory DemoOptions("Demo Options", "Demo control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(DemoOptions));
static cl::opt<int> amplifyFactor("f", cl::desc("Amplify factor"), cl::Required, cl::cat(DemoOptions));
static cl::opt<std::string> outputFile("o", cl::desc("Specify a file to save the modified .wav file."), cl::cat(DemoOptions));

typedef void (*PipelineFunctionType)(StreamSetPtr & ss_buf, void *buffer, size_t length);
PipelineFunctionType generatePipeline(CPUDriver &pxDriver, const unsigned int& amplifyFactor, const unsigned int &numChannels, const unsigned int &bitsPerSample)
{
    StreamSet * OutputBytes = pxDriver.CreateStreamSet(1,bitsPerSample * numChannels);

    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipelineWithIO({}, {Bind("OutputBytes", OutputBytes, ReturnedBuffer(1))}, 
                                             {Binding{Type::getInt8PtrTy(b.getContext()), "buffer"}, Binding{b.getSizeTy(), "length"}});
    Scalar * const buffer = P->getInputScalar("buffer");
    Scalar * const length = P->getInputScalar("length");

    StreamSet *dataStreams;
    ParseAudioBuffer(P, buffer, length, numChannels, bitsPerSample, dataStreams);
    
    std::vector<StreamSet *> OutputStreams(numChannels);

    for (unsigned i = 0; i < numChannels; ++i)
    {
        StreamSet *Channel = P->CreateStreamSet(1, bitsPerSample);
        StreamSet *BasisBits = P->CreateStreamSet(bitsPerSample);

        P->CreateKernelCall<IStreamSelect>(Channel, Select(dataStreams, {(unsigned)i}));
        S2P(P, bitsPerSample, Channel, BasisBits);
        //SHOW_BIXNUM(BasisBits);
        StreamSet *AmplifiedBasisBits = P->CreateStreamSet(bitsPerSample);
        P->CreateKernelCall<AmplifyPabloKernel>(bitsPerSample, BasisBits, amplifyFactor, AmplifiedBasisBits);
        //SHOW_STREAM(AmplifiedBasisBits);

        OutputStreams[i] = P->CreateStreamSet(1, bitsPerSample);
        P2S(P, AmplifiedBasisBits, OutputStreams[i]);
        //SHOW_BYTES(OutputStreams[i]);
    }
    
    P->CreateKernelCall<MergeKernel>(bitsPerSample, OutputStreams[0], OutputStreams[1], OutputBytes);
    SHOW_BYTES(OutputBytes);
    return reinterpret_cast<PipelineFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 2, bitsPerSample = 8, numSamples = 0;
    std::vector<int8_t, AlignedAllocator<int8_t,64>> buffer;
    bool isWav = true;
    try
    {
        readWAVFile(fd, numChannels, sampleRate, bitsPerSample, numSamples, buffer);
        std::cout << numChannels << " " << sampleRate << " " << bitsPerSample << " " << numSamples << "\n";
    }
    catch (const std::exception &e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        readTextFile(fd, buffer);
        numSamples = buffer.size() / (numChannels * (bitsPerSample / 8));
        isWav = false;
    }

    auto fn = generatePipeline(driver, amplifyFactor, numChannels, bitsPerSample);
    StreamSetPtr wavStream;

    fn(wavStream, &buffer[0], numSamples);
    if (outputFile.getNumOccurrences() != 0) {
        const int fd_out = open(outputFile.c_str(), O_WRONLY | O_CREAT, 0666);
        if (LLVM_UNLIKELY(fd_out == -1)) {
            llvm::errs() << "Error: cannot write to " << outputFile << ".\n";
        } else {
            if (isWav) {
                auto header = createWAVHeader(numChannels, sampleRate, bitsPerSample, numSamples);
                write(fd_out, header.c_str(), header.size());
            }
            // NOTE: Despite a sample can be 8, 16, 32, etc. we treat the stream as bytestream (8-bit) to make it consistent with existing kernels.
            write(fd_out, wavStream.data<8>(), wavStream.length() * numChannels * (bitsPerSample / 8));
            close(fd_out);
        }
    }
    close(fd);
    return 0;
}