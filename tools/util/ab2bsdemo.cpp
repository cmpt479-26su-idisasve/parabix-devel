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
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/streamutils/string_insert.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <audio/audio.h>
#include <iostream>

using namespace kernel;
using namespace llvm;
using namespace pablo;
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

static cl::OptionCategory Audio2BitStreamOptions("Audio2BitStream Options", "Audio2BitStream control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(Audio2BitStreamOptions));

typedef void (*AmplifierFunctionType)(uint32_t fd);
AmplifierFunctionType generatePipeline(CPUDriver &pxDriver, const bool& includedHeader, const unsigned int& numChannels, const unsigned int& numSamples, const unsigned int& sampleRate, const unsigned int& bitsPerSample)
{
    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}}, {});
    Scalar *fileDescriptor = P->getInputScalar("inputFileDecriptor");

    StreamSet *dataStreams;
    ExtractWAVData(P, fileDescriptor, numChannels, numSamples, sampleRate, bitsPerSample, includedHeader, dataStreams);
    SHOW_BIXNUM(dataStreams);
    return reinterpret_cast<AmplifierFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&Audio2BitStreamOptions, codegen::codegen_flags()});

    CPUDriver driver("audio2bitstream");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate, numChannels, bitsPerSample, numSamples;
    try
    {
        readWAVHeader(fd, numChannels, numSamples, sampleRate, bitsPerSample); 
        std::cout << numChannels << " " << numChannels << " " << sampleRate << " " << bitsPerSample << "\n";
        auto fn = generatePipeline(driver, false, numChannels, numSamples, sampleRate, bitsPerSample);
        fn(fd);
    }
    catch(const std::exception& e)
    {
        llvm::errs() << "Error: cannot read " << inputFile << " header for processing. Skipped.\n";
    }
    close(fd);
    return 0;
}