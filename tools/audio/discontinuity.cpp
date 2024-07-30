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
static cl::opt<int> threshold("t", cl::desc("Difference threshold"), cl::Required, cl::cat(DemoOptions));

typedef void (*PipelineFunctionType)(int32_t fd);
PipelineFunctionType generatePipeline(CPUDriver &pxDriver, const unsigned int& threshold, const unsigned int &numChannels, const unsigned int &bitsPerSample)
{
    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}});
    Scalar *fileDescriptor = P->getInputScalar("inputFileDecriptor");

    StreamSet *dataStreams;
    ParseAudioBuffer(P, fileDescriptor, numChannels, bitsPerSample, dataStreams);

    std::vector<StreamSet *> OutputStreams(numChannels);

    for (unsigned i = 0; i < numChannels; ++i)
    {
        StreamSet *Channel = P->CreateStreamSet(1, bitsPerSample);
        StreamSet *BasisBits = P->CreateStreamSet(bitsPerSample);

        P->CreateKernelCall<IStreamSelect>(Channel, Select(dataStreams, {(unsigned)i}));
        S2P(P, bitsPerSample, Channel, BasisBits);
        //SHOW_BIXNUM(BasisBits);
        StreamSet *MarkerStream = P->CreateStreamSet(1);
        P->CreateKernelCall<DiscontinuityKernel>(BasisBits, threshold, MarkerStream);
        
        SHOW_STREAM(MarkerStream);
    }
    //P->CreateKernelCall<StdOutKernel>(dataStreams);
    return reinterpret_cast<PipelineFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 1, bitsPerSample = 8, numSamples = 0;
    std::vector<int8_t, AlignedAllocator<int8_t,64>> buffer;
    try
    {
        readWAVFile(fd, numChannels, sampleRate, bitsPerSample, numSamples, buffer);
        lseek(fd, 44, SEEK_SET);
        std::cout << numChannels << " " << sampleRate << " " << bitsPerSample << " " << numSamples << "\n";
    }
    catch (const std::exception &e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        lseek(fd, 0, SEEK_SET);
        numSamples = buffer.size() / (numChannels * (bitsPerSample / 8));
    }

    auto fn = generatePipeline(driver, threshold, numChannels, bitsPerSample);
    fn(fd);
    close(fd);
    return 0;
}