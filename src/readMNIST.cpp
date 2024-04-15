#include "../include/readMNIST.hpp"

// Function to read MNIST images file and return vector of image data
std::vector<std::vector<double>> readMNISTImages(const std::string& imagePath) {
    std::vector<std::vector<double>> images;

    // Open the gzip-compressed images file
    gzFile file = gzopen(imagePath.c_str(), "rb");
    if (file == NULL) {
        throw std::runtime_error("Error: Failed to open images file");
    }

    // Read MNIST file header (magic number and metadata)
    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;
    // Read magic number and metadata with big-endian byte order
    gzread(file, &magicNumber, sizeof(magicNumber));
    gzread(file, &numImages, sizeof(numImages));
    gzread(file, &numRows, sizeof(numRows));
    gzread(file, &numCols, sizeof(numCols));

    magicNumber = __builtin_bswap32(magicNumber); // Convert to host byte order
    numImages = __builtin_bswap32(numImages); // Convert to host byte order
    numRows = __builtin_bswap32(numRows); // Convert to host byte order
    numCols = __builtin_bswap32(numCols); // Convert to host byte order

    // Validate magic number and image dimensions
    if (magicNumber != 0x00000803 || numRows != 28 || numCols != 28) {
        gzclose(file);
        
        // Print information when validation fails
        std::cout << "Invalid images file format:" << std::endl;
        std::cout << "Magic Number: 0x" << std::hex << magicNumber << std::dec << std::endl;
        std::cout << "Num Images: " << numImages << std::endl;
        std::cout << "Num Rows: " << numRows << std::endl;
        std::cout << "Num Cols: " << numCols << std::endl;

        throw std::runtime_error("Invalid images file format");
    }

    // Read image data
//     for (size_t i = 0; i < numImages; ++i) {
//         // std::vector<double> image(numRows * numCols);
//         std::vector<uint8_t> image(numRows * numCols);
// 
//         // Read raw image data (single channel, grayscale)
//         gzread(file, image.data(), sizeof(uint8_t) * numRows * numCols);
// 
//         // Normalize pixel values to range [0, 1] (assuming 8-bit grayscale)
//         for (uint8_t& pixel : image) {
//             pixel /= 255.0;
//         }
// 
//         // Add normalized image data to images vector
//         images.push_back(image);
//     }

    // Read image data
    for (size_t i = 0; i < numImages; ++i) {
        std::vector<uint8_t> rawImage(numRows * numCols);

        // Read raw image data (single channel, grayscale)
        gzread(file, rawImage.data(), sizeof(uint8_t) * numRows * numCols);

        // Normalize pixel values to range [0, 1] (assuming 8-bit grayscale)
        std::vector<double> normalizedImage(rawImage.size());
        for (size_t j = 0; j < rawImage.size(); ++j) {
            normalizedImage[j] = static_cast<double>(rawImage[j]) / 255.0;
        }

        // Add normalized image data to images vector
        images.push_back(normalizedImage);
    }

    // Close the file
    gzclose(file);

    return images;
}


// Function to read MNIST labels file and return vector of label data
std::vector<uint8_t> readMNISTLabels(const std::string& labelPath) {
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

    return labels;
}