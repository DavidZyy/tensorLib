#include "Tensor.hpp"
#include "readCSV.hpp"
#include "nn/modules.hpp"
#include "readMNIST.hpp"
#include <cstddef>
#include "iostream"


std::string testImgPath = "../dataset/MNIST/raw/t10k-images-idx3-ubyte.gz";
std::string testLabelsPath = "../dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz";

std::string trainImgPath = "../dataset/MNIST/raw/train-images-idx3-ubyte.gz";
std::string trainLabelsPath = "../dataset/MNIST/raw/train-labels-idx1-ubyte.gz";


const std::string csvFilePath = "../weights/fc_weight.csv";

int main() {
    Tensor<float> csvData = readCSV<float>(csvFilePath);
    Tensor<int> csvData_q = csvData.quantize();

    nn::Linear<int> fc1(csvData_q.shape()[1], csvData_q.shape()[0], std::move(csvData_q));
    // nn::Linear<float> fc1(csvData.shape()[1], csvData.shape()[0], std::move(csvData));

    Tensor<float> X_te = readMNISTImages<float>(testImgPath);
    Tensor<int> X_te_q = X_te.quantize();


    Tensor<int> result = fc1.forward(X_te_q);
    // Tensor<float> result = fc1.forward(X_te);

    Tensor<int> label = readMNISTLabels<int>(testLabelsPath);

    Tensor<int> pred = result.argmax(1);

    // std::cout << pred << std::endl;

    Tensor<int> correct = (pred == label);

    auto meanValue = correct.mean();

    std::cout << meanValue << std::endl;
}
