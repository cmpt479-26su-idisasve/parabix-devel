#include "audio/audio.h"
#include <iostream>
#include <kernel/io/source_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Value.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/core/relationship.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/streamutils/deletion.h>
#include "audio/stream_manipulation.h"

#define SHOW_STREAM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBixNum(#name, name)
#define SHOW_BYTES(name)            \
    if (codegen::EnableIllustrator) \
    P->captureByteData(#name, name)

#define NUM_HEADER_BYTES 44

namespace audio
{
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

        StreamSet *TrimByteStream;
        if (includedHeader)
        {
            StreamSet *BitStreams = P->CreateStreamSet(8);
            P->CreateKernelCall<S2PKernel>(ByteStream, BitStreams);
            StreamSet *ones = P->CreateStreamSet(1);
            StreamSet *shiftedOnes = P->CreateStreamSet(1);
            P->CreateKernelCall<CreateOnes>(BitStreams, ones);
            P->CreateKernelCall<ShiftBack>(ones, shiftedOnes, NUM_HEADER_BYTES);
            StreamSet *headerMask = P->CreateStreamSet(1);
            P->CreateKernelCall<ShiftForward>(shiftedOnes, headerMask, NUM_HEADER_BYTES);
            StreamSet *TrimBitStreams = P->CreateStreamSet(8);
            FilterByMask(P, headerMask, BitStreams, TrimBitStreams);
            TrimByteStream = P->CreateStreamSet(1, 8);
            P->CreateKernelCall<P2SKernel>(TrimBitStreams, TrimByteStream);
        }
        else
        {
            TrimByteStream = ByteStream;
        }
        
        SHOW_BYTES(TrimByteStream);
        StreamSet *DataStreams = P->CreateStreamSet(numChannels, 8);
        if (numChannels == 2)
        {
            P->CreateKernelCall<mS2PKernel>(TrimByteStream, DataStreams, bitPerSample);
        }
        else
        {
            DataStreams = ByteStream;
        }
        outputDataStreams = DataStreams;
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

#undef NUM_HEADER_BYTES