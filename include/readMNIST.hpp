#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "zlib.h" // For decompression of gzip files

// Function to read MNIST images file and return vector of image data
std::vector<std::vector<double>> readMNISTImages(const std::string& imagePath);

std::vector<uint8_t> readMNISTLabels(const std::string& labelPath);
