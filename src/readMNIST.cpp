#include "../include/readMNIST.hpp"

Tensor<uint8_t> readMNISTLabels(const std::string& labelPath) {
    std::vector<uint8_t> labels;

    // Open the gzip-compressed labels file
    gzFile file = gzopen(labelPath.c_str(), "rb");
    if (file == NULL) {
        throw std::runtime_error("Error: Failed to open labels file");
    }

    // Read MNIST file header (magic number and metadata)
    uint32_t magicNumber;
    uint32_t numLabels;
    gzread(file, &magicNumber, sizeof(magicNumber));
    gzread(file, &numLabels, sizeof(numLabels));
    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    // Validate magic number
    if (magicNumber != 0x00000801) {
        gzclose(file);
        throw std::runtime_error("Error: Invalid labels file format");
    }

    // Read label data
    labels.resize(numLabels);
    gzread(file, labels.data(), sizeof(uint8_t) * numLabels);

    // Close the file
    gzclose(file);

    std::vector<int> shape = {static_cast<int>(labels.size())};
    Tensor<uint8_t> tensor(shape);

    // Copy the data from the vector of vectors to the Tensor
    for (size_t i = 0; i < labels.size(); ++i) {
        std::vector<int> indices = {static_cast<int>(i)};
        tensor(indices) = labels[i];
    }

    return tensor;
}
