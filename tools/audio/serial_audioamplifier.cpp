#include <iostream>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <unistd.h>
#include <vector>
#include <string>
#include <endian.h> // for htole16

void processing(const int &fd,
                       unsigned int &numChannels,
                       unsigned int &sampleRate,
                       unsigned int &bitsPerSample,
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
        bitsPerSample = bits_per_sample;

        bytesRead = read(fd, temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "data")
        {
            throw std::runtime_error("Error parsing file format: Subchunk 2 ID does not match Wav format.");
        }

        unsigned subchunk2_size;
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

        numSamples = subchunk2_size / (numChannels * bitsPerSample / 8);

        const int fd_out = open("out.wav", O_WRONLY | O_CREAT, 0666);

        std::vector<int32_t> first(numSamples);
        std::vector<int32_t> second(numSamples);
        for (int i=0;i<first.size();++i)
        {   
            int16_t firstValue = (static_cast<int16_t>(data_buffer[4*i]) << 8) | data_buffer[4*i + 1];
            int16_t secondValue = (static_cast<int16_t>(data_buffer[4*i+2]) << 8) | data_buffer[4*i + 3];
            first[i] = firstValue * 2;
            second[i] = secondValue * 2;
            if (first[i] > (1<<15)-1)
            {
                first[i] = (1<<15)-1;
            }
            else if (first[i] < -(1<<15))
            {
                first[i] = -(1<<15);
            }

            if (second[i] > (1<<15)-1)
            {
                second[i] = (1<<15)-1;
            } 
            else if (second[i] < -(1<<15))
            {
                second[i] = -(1<<15);
            }
        }

        std::vector<int16_t> merged(numSamples*2);
        for (int i=0;i<first.size();++i)
        {
            merged[2*i] = first[i];
            merged[2*i+1] = second[i];
        }

        write(fd_out, merged.data(), merged.size() *sizeof(int16_t));
    }

int main(int argc, char *argv[])
{
    unsigned int numChannels, sampleRate, bitsPerSample, numSamples;
    const int fd = open(argv[1], O_RDONLY);
    processing(fd, numChannels, sampleRate, bitsPerSample, numSamples);
    

}