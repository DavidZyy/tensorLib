#include "readMNIST.hpp"


int main() {
    try {
        // Specify paths to MNIST dataset files
        std::string imagesPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-images-idx3-ubyte.gz";
        std::string labelsPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-labels-idx1-ubyte.gz";

        // Read MNIST images and labels
        Tensor<float> X_tr = readMNISTImages<float>(imagesPath);
        Tensor<int> y_tr = readMNISTLabels<int>(labelsPath);

        // Display information about the loaded data
        std::cout << "Number of images: " << X_tr.shape()[0] << std::endl;
        std::cout << "Image dimensions: " << X_tr.shape()[1] << std::endl;
        std::cout << "Number of labels: " << y_tr.shape()[0] << std::endl;

        // Access and process the loaded image and label data as needed
        // Example: Print the first label and first few pixels of the first image
        std::cout << "First label: " << static_cast<int>(y_tr({10})) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

