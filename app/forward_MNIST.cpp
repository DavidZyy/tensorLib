#include "../include/Tensor.hpp"
#include "../include/readCSV.hpp"
#include "../include/nn/modules.hpp"
#include "../include/readMNIST.hpp"
#include <cstddef>
#include "iostream"

std::string testImgPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/t10k-images-idx3-ubyte.gz";
std::string trainImgPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-images-idx3-ubyte.gz";
std::string trainLabelsPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-labels-idx1-ubyte.gz";
std::string testLabelsPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/t10k-labels-idx1-ubyte.gz";

const std::string csvFilePath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/src/theta.csv";

int main() {
    Tensor<float> csvData = readCSV<float>(csvFilePath);

    nn::Linear<float> fc1(csvData.shape()[0], csvData.shape()[1], std::move(csvData));

    Tensor<float> X_te = readMNISTImages<float>(testImgPath);

    Tensor<float> result = fc1.forward(X_te);

    Tensor<int> label = readMNISTLabels<int>(testLabelsPath);

    Tensor<int> pred = result.argmax(1);

    // std::cout << pred << std::endl;

    Tensor<int> correct = pred == label;

    auto meanValue = correct.mean();

    std::cout << meanValue << std::endl;
}
