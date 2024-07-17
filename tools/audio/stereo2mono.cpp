#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <string>
#include <toolchain/toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <audio/audio.h>
#include <iostream>

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
static cl::opt<std::string> outputFile("o", cl::desc("Specify a file to save the modified .wav file."), cl::cat(DemoOptions));

typedef void (*PipelineFunctionType)(StreamSetPtr & ss_buf, uint32_t fd);
PipelineFunctionType generatePipeline(CPUDriver &pxDriver, const unsigned int& numChannels, const unsigned int& numSamples, const unsigned int& bitsPerSample, const unsigned int& sampleRate, const bool& isWav)
{
    StreamSet * OutputBytes = pxDriver.CreateStreamSet(1, bitsPerSample);

    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipelineWithIO({}, {Bind("OutputBytes", OutputBytes, ReturnedBuffer(1))}, 
                                             {Binding{b.getInt32Ty(), "inputFileDecriptor"}});
    Scalar *fileDescriptor = P->getInputScalar("inputFileDecriptor");

    StreamSet *dataStreams;
    ExtractWAVData(P, fileDescriptor, numChannels, numSamples, sampleRate, bitsPerSample, /*trim_header*/ isWav, dataStreams);
    SHOW_BYTES(dataStreams);

    StreamSet *FirstChannelStream = P->CreateStreamSet(1, 8);
    StreamSet *FirstChannelBasisBits = P->CreateStreamSet(bitsPerSample);
    P->CreateKernelCall<IStreamSelect>(FirstChannelStream, Select(dataStreams, {(unsigned)0}));
    S2P(P, bitsPerSample, FirstChannelStream, FirstChannelBasisBits);
    //SHOW_BIXNUM(FirstChannelBasisBits[i]);

    StreamSet *SecondChannelStream = P->CreateStreamSet(1, 8);
    StreamSet *SecondChannelBasisBits = P->CreateStreamSet(bitsPerSample);
    P->CreateKernelCall<IStreamSelect>(SecondChannelStream, Select(dataStreams, {(unsigned)1}));
    S2P(P, bitsPerSample, SecondChannelStream, SecondChannelBasisBits);
    //SHOW_BIXNUM(OutputStreams[i]);

    StreamSet *MonoBasisBits = P->CreateStreamSet(bitsPerSample);
    P->CreateKernelCall<Stereo2MonoPabloKernel>(FirstChannelBasisBits, SecondChannelBasisBits, MonoBasisBits);   

    P2S(P, MonoBasisBits, OutputBytes);
    SHOW_BYTES(OutputBytes);

    return reinterpret_cast<PipelineFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate=0, numChannels=2, bitsPerSample=16, numSamples=0;
    bool isWav = true;
    try
    {
        readWAVHeader(fd, numChannels, numSamples, bitsPerSample, sampleRate); 
        std::cout << numChannels << " " << numChannels << " " << sampleRate << " " << bitsPerSample << "\n";
    }
    catch(const std::exception& e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        isWav = false;
    }

    auto fn = generatePipeline(driver, numChannels, numSamples, bitsPerSample, sampleRate, isWav);
    StreamSetPtr outputStream;
    fn(outputStream, fd);

    if (outputFile.getNumOccurrences() != 0) {
        const int fd_out = open(outputFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (LLVM_UNLIKELY(fd_out == -1)) {
            llvm::errs() << "Error: cannot write to " << outputFile << ".\n";
        } else {
            if (isWav) {
                auto header = createWAVHeader(1, sampleRate, bitsPerSample, numSamples);
                write(fd_out, header.c_str(), header.size());
            }
            // NOTE: Despite a sample can be 8, 16, 32, etc. we treat the stream as bytestream (8-bit) to make it consistent with existing kernels.
            write(fd_out, outputStream.data<8>(), outputStream.length());
            close(fd_out);
        }
    }
    close(fd);
    return 0;
}