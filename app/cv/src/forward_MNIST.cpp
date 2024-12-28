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

const std::string fc_weight1Path = "../weights/fc_weight1.csv";
const std::string fc_weight2Path = "../weights/fc_weight2.csv";
const std::string fc_weight3Path = "../weights/fc_weight3.csv";

template <typename dtype>
class CNN {
public:
    CNN () : fc1(784, 10), fc2(10, 784), fc3(784, 10) {
    };

    ~ CNN() {};

    Tensor<dtype> forward(const Tensor<float>& input) {
        return fc3.forward(fc2.forward(fc1.forward(input)));
    }

// private:

    nn::Linear<dtype> fc1;
    nn::Linear<dtype> fc2;
    nn::Linear<dtype> fc3;
};

template <typename dtype>
class CNN1 {
public:
    CNN1 () : fc1(784, 10) {
    };

    ~ CNN1() {};

    Tensor<dtype> forward(const Tensor<float>& input) {
        return fc1.forward(input);
    }

// private:

    nn::Linear<dtype> fc1;
};

int main() {
    auto model = CNN1<float>();

    Tensor<float> fc_weight1 = readCSV<float>(fc_weight1Path);
    // Tensor<float> fc_weight2 = readCSV<float>(fc_weight2Path);
    // Tensor<float> fc_weight3 = readCSV<float>(fc_weight3Path);

    model.fc1.weight = std::move(fc_weight1);
    // model.fc2.weight = std::move(fc_weight2);
    // model.fc3.weight = std::move(fc_weight3);

    Tensor<float> X_te = readMNISTImages<float>(testImgPath);

    Tensor<float> result = model.forward(X_te);

    Tensor<int> label = readMNISTLabels<int>(testLabelsPath);

    Tensor<int> pred = result.argmax(1);

    Tensor<int> correct = (pred == label);

    // convert int to float for mean
    auto correct2 = static_cast<Tensor<float>>(correct);

    // auto meanValue = correct.mean(0);
    auto meanValue = correct2.mean(0);

    std::cout << meanValue << std::endl;
}
