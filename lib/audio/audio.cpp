#include "audio/audio.h"

Audio Audio::fromWAV(const std::string & file_path)
{
    Audio audio;

    std::ifstream file(file_path, std::ios_base::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    char temp_buffer[11];

    // validate file format
    file.read(temp_buffer, 8);
    temp_buffer[4] = '\0';
    if (!file || std::string(temp_buffer) != "RIFF")
    {
        throw std::runtime_error("Error parsing file format: Chunk ID does not match Wav format.");
    }

    file.read(temp_buffer, 4);
    temp_buffer[4] = '\0';
    if (!file || std::string(temp_buffer) != "WAVE")
    {
        throw std::runtime_error("Error parsing file format: Header does not match Wav format.");
    }

    file.read(temp_buffer, 10);
    temp_buffer[4] = '\0';
    if (!file || std::string(temp_buffer) != "fmt\x20")
    {
        throw std::runtime_error("Error parsing file format: Subchunk ID does not match Wav format.");
    }

    file.read(reinterpret_cast<char *>(&audio.num_channels), 2);
    if (!file)
    {
        throw std::runtime_error("Error parsing file format: Cannot interpret number channels.");
    }

    file.read(reinterpret_cast<char*>(&audio.sample_rate), 4);
    if (!file)
    {
        throw std::runtime_error("Error parsing file format: Cannot interpret sample rate.");
    }

    // assume bits per sample is 16, no need to read it. TODO: handle arbitrary number of bits per sample
    file.read(temp_buffer, 8);
    if (!file)
    {
        throw std::runtime_error("Error parsing file format.");
    }

    file.read(temp_buffer, 4);
    temp_buffer[4] = '\0';
    if (!file || std::string(temp_buffer) != "data")
    {
        throw std::runtime_error("Error parsing file format: Subchunk 2 ID does not match Wav format.");
    }

    int subchunk2_size;
    file.read(reinterpret_cast<char*>(&subchunk2_size), 4);
    if (!file)
    {
        throw std::runtime_error("Error parsing file format: Cannot interpret subchunk 2 size.");
    }

    // copy over the data buffer
    audio.data_buffer.clear();
    audio.data_buffer.resize(subchunk2_size);
    file.read(reinterpret_cast<char*>(&audio.data_buffer[0]), subchunk2_size);

    return audio;
}
