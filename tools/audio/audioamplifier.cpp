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

typedef void (*PipelineFunctionType)(uint32_t fd);
PipelineFunctionType generatePipeline(CPUDriver &pxDriver, const unsigned int &numChannels, const unsigned int &numSamples, const unsigned int &bitsPerSample, const unsigned int &sampleRate, const bool &isWav)
{
    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}}, {});
    Scalar *fileDescriptor = P->getInputScalar("inputFileDecriptor");

    StreamSet *dataStreams;
    ExtractWAVData(P, fileDescriptor, numChannels, numSamples, sampleRate, bitsPerSample, /*trim_header*/ isWav, dataStreams);
    SHOW_BYTES(dataStreams);
    
    std::vector<StreamSet *> OutputStreams(numChannels);

    for (int i = 0; i < numChannels; ++i)
    {
        
        StreamSet *Channel = P->CreateStreamSet(1, 8);
        StreamSet *BasisBits = P->CreateStreamSet(bitsPerSample);

        P->CreateKernelCall<IStreamSelect>(Channel, Select(dataStreams, {i}));
        S2P(P, bitsPerSample, Channel, BasisBits);
        //SHOW_STREAM(BasisBits);
        StreamSet *AmplifiedBasisBits = P->CreateStreamSet(bitsPerSample);
        P->CreateKernelCall<AmplifyPabloKernel>(bitsPerSample, BasisBits, 2, AmplifiedBasisBits);
        //SHOW_STREAM(AmplifiedBasisBits);

        OutputStreams[i] = P->CreateStreamSet(1, 16);
        P2S(P, AmplifiedBasisBits, OutputStreams[i]);
        SHOW_BYTES(OutputStreams[i]);
    }
    
    StreamSet *outputDataStream = P->CreateStreamSet(1, 16);
    P->CreateKernelCall<MergeKernel>(16, OutputStreams[0], OutputStreams[1], outputDataStream);
    SHOW_BYTES(outputDataStream);
    return reinterpret_cast<PipelineFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 2, bitsPerSample = 16, numSamples = 0;
    bool isWav = true;
    try
    {
        readWAVHeader(fd, numChannels, numSamples, bitsPerSample, sampleRate);
        std::cout << numChannels << " " << numChannels << " " << sampleRate << " " << bitsPerSample << "\n";
    }
    catch (const std::exception &e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        isWav = false;
    }

    auto fn = generatePipeline(driver, numChannels, numSamples, bitsPerSample, sampleRate, isWav);
    fn(fd);
    close(fd);
    return 0;
}