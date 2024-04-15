#include "../include/readMNIST.hpp"


int main() {
    try {
        // Specify paths to MNIST dataset files
        std::string imagesPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-images-idx3-ubyte.gz";
        std::string labelsPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-labels-idx1-ubyte.gz";

        // Read MNIST images and labels
        std::vector<std::vector<double>> X_tr = readMNISTImages(imagesPath);
        std::vector<uint8_t> y_tr = readMNISTLabels(labelsPath);

        // Display information about the loaded data
        std::cout << "Number of images: " << X_tr.size() << std::endl;
        std::cout << "Image dimensions: " << X_tr[0].size() << std::endl;
        std::cout << "Number of labels: " << y_tr.size() << std::endl;

        // Access and process the loaded image and label data as needed
        // Example: Print the first label and first few pixels of the first image
        std::cout << "First label: " << static_cast<int>(y_tr[10]) << std::endl;
        std::cout << "First few pixels of first image:" << std::endl;
        for (size_t i = 0; i < X_tr.size(); ++i) {
            std::cout << X_tr[0][i] << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

