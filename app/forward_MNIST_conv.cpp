#include "../include/Tensor.hpp"
#include "../include/readCSV.hpp"
#include "../include/nn/modules.hpp"
#include "../include/readMNIST.hpp"
#include <cstddef>
#include "iostream"
#include <chrono>

std::string testImgPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/t10k-images-idx3-ubyte.gz";
std::string testLabelsPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/t10k-labels-idx1-ubyte.gz";

std::string trainImgPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-images-idx3-ubyte.gz";
std::string trainLabelsPath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/data/train-labels-idx1-ubyte.gz";

const std::string conv1WeightPath = "/home/zhuyangyang/project/test/conv1_weight.csv";
const std::string fcWeightPath = "/home/zhuyangyang/project/test/fc_weight.csv";

int main() {

    Tensor<float> conv1Weight = readCSV<float>(conv1WeightPath);
    conv1Weight = conv1Weight.view({1,1,3,3,});
    nn::Conv2d<float> conv1(1,1,3,1,1,std::move(conv1Weight));

    Tensor<float> fcWeight = readCSV<float>(fcWeightPath);
    nn::Linear<float> fc1(fcWeight.shape()[1], fcWeight.shape()[0], std::move(fcWeight));

    Tensor<float> X_te = readMNISTImages<float>(testImgPath);
    X_te = X_te.view({10000, 1,28, 28});

    int slice_N = 1000;
    X_te = X_te.slice(0, slice_N, 0);

    Tensor<float> result1 = conv1.forward(X_te);
    // result1 = result1.view({10000, 28 * 28});
    result1 = result1.view({slice_N, 28 * 28});

    Tensor<float> result2 = fc1.forward(result1);


    Tensor<int> label = readMNISTLabels<int>(testLabelsPath);

    label = label.slice(0, slice_N, 0);

    Tensor<int> pred = result2.argmax(1);

    // std::cout << pred << std::endl;

    Tensor<int> correct = (pred == label);

    auto meanValue = correct.mean();

    std::cout << meanValue << std::endl;

    return 0;
}
