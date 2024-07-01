#include "audio/audio.h"
#include <iostream>
#include <kernel/io/source_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Value.h>
#include <kernel/core/relationship.h>

using namespace kernel;
using namespace llvm;

namespace audio
{
    mS2PKernel::mS2PKernel(KernelBuilder &b, StreamSet *const inputStreams, StreamSet *const outputStreams, const unsigned int bitsPerSample)
        : MultiBlockKernel(b, "mS2PKernel_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(bitsPerSample),
                           {Binding{"inputStreams", inputStreams, FixedRate(2)}},
                           {Binding{"outputStreams", outputStreams, FixedRate(1)}}, {}, {}, {}),
          numInputStreams(inputStreams->getNumElements()), bitsPerSample(bitsPerSample) {}

    void mS2PKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = 8;
        const unsigned inputPacksPerStride = fw * 2;
        const unsigned outputPacksPerStride = fw * 1;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *packLoop = b.CreateBasicBlock("packLoop");
        BasicBlock *packFinalize = b.CreateBasicBlock("packFinalize");
        Constant *const ZERO = b.getSize(0);
        Value *numOfBlocks = numOfStrides;
        b.CreateBr(packLoop);
        b.SetInsertPoint(packLoop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        for (int streamIndex = 0; streamIndex < numInputStreams; ++streamIndex)
        {
            Value *bytepack[inputPacksPerStride];
            Constant *const STREAMINDEX = b.getSize(streamIndex);
            Constant *const LOWSTREAMINDEX = b.getSize(2 * streamIndex);
            Constant *const HIGHSTREAMINDEX = b.getSize(2 * streamIndex + 1);
            for (unsigned i = 0; i < inputPacksPerStride; i++)
            {
                bytepack[i] = b.loadInputStreamPack("inputStreams", STREAMINDEX, b.getInt32(i), blockOffsetPhi);
            }

            Value *lo[outputPacksPerStride];
            Value *hi[outputPacksPerStride];
            for (unsigned i = 0; i < outputPacksPerStride; i++)
            {
                lo[i] = b.hsimd_packl(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
                hi[i] = b.hsimd_packh(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
                b.storeOutputStreamPack("outputStreams", LOWSTREAMINDEX, b.getInt32(i), blockOffsetPhi, lo[i]);
                b.storeOutputStreamPack("outputStreams", HIGHSTREAMINDEX, b.getInt32(i), blockOffsetPhi, hi[i]);
            }
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, packLoop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);
        b.SetInsertPoint(packFinalize);
    }

    void ExtractWAVData(
        const std::unique_ptr<ProgramBuilder> &P,
        Scalar *const fileDescriptor,
        unsigned int numChannels,
        unsigned int numSamples,
        unsigned int sampleRate,
        unsigned int bitPerSample,
        const bool includedHeader,
        StreamSet *&outputDataStreams)
    {
        if (numChannels != 1 && numChannels != 2)
        {
            throw std::invalid_argument("Error: numChannels " + std::to_string(numChannels) + " is not valid");
        }

        StreamSet *ByteStream = P->CreateStreamSet(1, 8);
        P->CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
        StreamSet *DataStreams = P->CreateStreamSet(numChannels, 8);
        if (numChannels == 2)
        {
            P->CreateKernelCall<mS2PKernel>(ByteStream, DataStreams, bitPerSample);
        }
        else
        {
            DataStreams = ByteStream;
        }
    }

    void readWAVHeader(const int& fd,
                       unsigned int &numChannels,
                       unsigned int &sampleRate,
                       unsigned int &bitPerSample,
                       unsigned int &numSamples)
    {
        char temp_buffer[11];

        // validate file format
        ssize_t bytesRead = read(fd, &temp_buffer, 8);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "RIFF")
        {
            throw std::runtime_error("Error parsing file format: Chunk ID does not match Wav format.");
        }

        bytesRead = read(fd, &temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "WAVE")
        {
            throw std::runtime_error("Error parsing file format: Header does not match Wav format.");
        }

        bytesRead = read(fd, &temp_buffer, 10);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "fmt\x20")
        {
            throw std::runtime_error("Error parsing file format: Subchunk ID does not match Wav format.");
        }

        // read 2 bytes for num channels
        uint16_t num_channels;
        bytesRead = read(fd, reinterpret_cast<char *>(&num_channels), 2);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret number channels.");
        }
        numChannels = num_channels;

        // read 4 bytes for sample rate
        bytesRead = read(fd, reinterpret_cast<char *>(&sampleRate), 4);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret sample rate.");
        }

        // skip the next 6 bytes
        bytesRead = read(fd, temp_buffer, 6);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format.");
        }

        // read 4 bytes for num bits per sample
        uint16_t bits_per_sample;
        bytesRead = read(fd, reinterpret_cast<char *>(&bits_per_sample), 2);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format.");
        }
        bitPerSample = bits_per_sample;

        bytesRead = read(fd, temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "data")
        {
            throw std::runtime_error("Error parsing file format: Subchunk 2 ID does not match Wav format.");
        }

        int subchunk2_size;
        bytesRead = read(fd, reinterpret_cast<char *>(&subchunk2_size), 4);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret subchunk 2 size.");
        }

        // copy over the data buffer
        std::vector<u_char> data_buffer(subchunk2_size);
        bytesRead = read(fd, reinterpret_cast<char *>(&data_buffer[0]), subchunk2_size);

        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret data chunk.");
        }

        numSamples = subchunk2_size / (numChannels * bitPerSample / 8);
    }
}
