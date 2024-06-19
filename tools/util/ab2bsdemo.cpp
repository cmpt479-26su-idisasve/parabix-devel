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

class mS2PKernel : public PabloKernel
{
public:
    mS2PKernel(KernelBuilder &b, StreamSet *const inputs, StreamSet *outputs, const std::string& suffix = "")
        : PabloKernel(b,
                      "S2PKernel_" + suffix,
                      {Binding{"inputs", inputs, FixedRate(2), LookAhead(2)}},
                      {Binding{"outputs", outputs}}) {}

protected:
    void generatePabloMethod() override
    {
        PabloBuilder pb(getEntryScope());
        std::vector<PabloAST *> inputs = getInputStreamSet("inputs");
        std::vector<PabloAST *> parallelized(inputs.size() * 2);
        PabloAST *const ones = pb.createOr(inputs[0], pb.createOnes());
        PabloAST *const odd_positions = pb.createEveryNth(ones, pb.getInteger(2));

        for (unsigned int i = 0; i < parallelized.size(); ++i)
        {
            if (i < inputs.size())
            {
                parallelized[i] = pb.createSel(odd_positions, inputs[i], pb.createLookahead(inputs[i], 1));
            }
            else
            {
                auto input_index = i - inputs.size();
                parallelized[i] = pb.createSel(odd_positions, pb.createLookahead(inputs[input_index], 1), pb.createLookahead(inputs[input_index], 2));
            }
        }

        Var *const outputs = getOutputStreamVar("outputs");
        for (unsigned i = 0; i < parallelized.size(); ++i)
        {
            pb.createAssign(pb.createExtract(outputs, pb.getInteger(i)), parallelized[i]);
        }
    }
};

typedef void (*AmplifierFunctionType)(uint32_t fd);
AmplifierFunctionType generatePipeline(CPUDriver &pxDriver, const bool& trim_header)
{
    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}}, {});
    Scalar *fileDescriptor = P->getInputScalar("inputFileDecriptor");
    StreamSet *ByteStream = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    StreamSet *BasisBits = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
    StreamSet * rawDataBasisBits = P->CreateStreamSet(8);
    if (trim_header)
    {
        P->CreateKernelCall<ShiftBack>(BasisBits, rawDataBasisBits, 44); // shift 1 for testing purpose, actual wav should be shifted by 44
    }
    else
    {
        rawDataBasisBits = BasisBits;
    }
    StreamSet *rawDataBasisBits16 = P->CreateStreamSet(16);
    P->CreateKernelCall<mS2PKernel>(rawDataBasisBits, rawDataBasisBits16, "_8->16");
    SHOW_BIXNUM(rawDataBasisBits16);

    StreamSet *rawDataBasisBits32 = P->CreateStreamSet(32);
    P->CreateKernelCall<mS2PKernel>(rawDataBasisBits16, rawDataBasisBits32, "_16->32");
    SHOW_BIXNUM(rawDataBasisBits32);
    return reinterpret_cast<AmplifierFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&Audio2BitStreamOptions, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    bool trim_header = true;
    try
    {
        Audio audio = Audio::fromWAV(inputFile);
        std::cout << "Sample rate: " << audio.sample_rate << " Hz\n";
        std::cout << "Num channels: " << audio.num_channels << " \n";
        std::cout << "Num bits per sample: " << audio.bits_per_sample << " \n";
        std::cout << "Num samples: " << audio.num_samples << " \n";
    }
    catch (const std::exception& e)
    {
        trim_header = false;
        std::cout << "Not a valid wave file, trim header is off.\n";
    }

    CPUDriver driver("audio2bitstream");
    auto fn = generatePipeline(driver, trim_header);
    const int fd = open(inputFile.c_str(), O_RDONLY);
    fn(fd);
    close(fd);
    return 0;
}