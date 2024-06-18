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
#include <kernel/basis/p2s_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <audio/audio.h>
#include <iostream>

using namespace kernel;
using namespace llvm;
using namespace pablo;

#define SHOW_STREAM(name) P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name)  P->captureBixNum(#name, name)
#define SHOW_BYTES(name) P->captureByteData(#name, name)

typedef void (*AmplifierFunctionType)(uint32_t fd);

AmplifierFunctionType generatePipeline(CPUDriver & pxDriver, const uint16_t& num_channels, const uint32_t& num_samples, const uint16_t& bits_per_sample) {
    auto & b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b.getInt32Ty(), "inputFileDecriptor"}}, {});
    Scalar * fileDescriptor = P->getInputScalar("inputFileDecriptor");
    StreamSet * ByteStream = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    StreamSet * BasisBits = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    StreamSet * NewBasisBits = P->CreateStreamSet(8);
    P->CreateKernelCall<ShiftBack>(BasisBits, NewBasisBits, 1);
    
    StreamSet * DataByteStream = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(NewBasisBits, DataByteStream);
    
    P->CreateKernelCall<StdOutKernel>(DataByteStream);
    return reinterpret_cast<AmplifierFunctionType>(P->compile());
}

int main()
{
    std::string inputFile = "audio1.wav";
    Audio audio = Audio::fromWAV(inputFile);
    std::cout << "Sample rate: " << audio.sample_rate << " Hz\n";
    std::cout << "Num channels: " << audio.num_channels << " \n";
    std::cout << "Num bits per sample: " << audio.bits_per_sample << " \n";
    std::cout << "Num samples: " << audio.num_samples << " \n";
    
    CPUDriver driver("amplifier");
    auto fn = generatePipeline(driver, audio.num_channels, audio.num_samples, audio.bits_per_sample);
    const int fd = open(inputFile.c_str(), O_RDONLY);
    fn(fd);
    close(fd);
    return 0;
}