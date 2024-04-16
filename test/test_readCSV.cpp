#include "../include/readCSV.hpp"
#include <iostream>
#include <iterator>
#include "../include/Tensor.hpp"

int main() {
    // Specify the path to the CSV file
    const std::string csvFilePath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/src/theta.csv";

    try {
        Tensor<float> csvData = readCSV<float>(csvFilePath);

        // std::cout<<"tensor address: "<<&csvData<<" tensor data address: "<<&csvData.data_<<std::endl;

        // Display the read data
        std::cout << "CSV data dimensions: ";
        
        for (size_t i = 0; i < csvData.shape().size(); ++i) {
            std::cout << csvData.shape()[i];
            if (i < csvData.shape().size() - 1) {
                std::cout << " x ";
            }
        }

        std::cout<<std::endl;
        // std::cout << csvData << std::endl; 

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
