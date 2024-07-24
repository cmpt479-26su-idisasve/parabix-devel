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

typedef void (*PipelineFunctionType)(uint32_t fd);
PipelineFunctionType generatePipeline(CPUDriver &pxDriver, const unsigned int& threshold, const unsigned int &numChannels, const unsigned int &bitsPerSample, const bool &isWav)
{
    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}});
    Scalar *fileDescriptor = P->getInputScalar("inputFileDecriptor");

    StreamSet *InputStream = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, InputStream);

    StreamSet *BitsBasis = P->CreateStreamSet(8);
    S2P(P, bitsPerSample, InputStream, BitsBasis);
    SHOW_STREAM(BitsBasis);

    StreamSet *MarkerStream = P->CreateStreamSet(1);
    P->CreateKernelCall<DiscontinuityKernel>(BitsBasis, threshold, MarkerStream);
    SHOW_STREAM(MarkerStream);

    P->CreateKernelCall<StdOutKernel>(MarkerStream);
    return reinterpret_cast<PipelineFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 2, bitsPerSample = 8, numSamples = 0;
    bool isWav = true;
    try
    {
        readWAVHeader(fd, numChannels, sampleRate, bitsPerSample, numSamples);
        std::cout << numChannels << " " << sampleRate << " " << bitsPerSample << " " << numSamples << "\n";
    }
    catch (const std::exception &e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        isWav = false;
    }

    auto fn = generatePipeline(driver, threshold, numChannels, bitsPerSample, isWav);
    fn(fd);
    close(fd);
    return 0;
}