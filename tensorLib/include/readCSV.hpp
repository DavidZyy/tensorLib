#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "Tensor.hpp"

template <typename dtype>
Tensor<dtype> readCSV(const std::string& filename) {
    std::vector<std::vector<dtype>> data;

    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Failed to open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<dtype> row;
        std::istringstream iss(line);
        std::string cell;

        while (std::getline(iss, cell, ',')) {
            // Convert string cell to dtype
            try {
                dtype value = static_cast<dtype>(std::stod(cell));
                row.push_back(value);
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Error: Invalid data format in CSV file");
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Error: Out of range data in CSV file");
            }
        }

        data.push_back(row);
    }

    // Close the file
    file.close();

    // Determine the shape of the tensor
    std::vector<int> shape = {static_cast<int>(data.size())};
    if (!data.empty()) {
        shape.push_back(static_cast<int>(data[0].size()));
    }

    // Create a Tensor object with the inferred shape
    Tensor<dtype> tensor(shape);

    // Copy the data from the vector of vectors to the Tensor
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            std::vector<int> indices = {static_cast<int>(i), static_cast<int>(j)};
            tensor(indices) = data[i][j];
        }
    }

    // std::cout<<"tensor address: "<<&tensor<<" tensor data address: "<<&tensor.data_<<std::endl;

    // return std::move(tensor);
    return tensor;
}
