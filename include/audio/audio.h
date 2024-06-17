#pragma once
#include <vector>
#include <fstream>
#include <string>

class Audio {
public:
    static Audio fromWAV(const std::string& file_path);

private:
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<unsigned char> data_buffer = std::vector<unsigned char>();
};