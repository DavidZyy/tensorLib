#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

// Function to read CSV file into a vector of vectors of doubles
std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;

    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Failed to open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        std::string cell;

        while (std::getline(iss, cell, ',')) {
            // Convert string cell to double
            try {
                double value = std::stod(cell);
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

    // std::cout<< "in  func" << &data << std::endl;

    return data;
}

int main() {
    // Specify the path to the CSV file
    std::string csvFilePath = "/home/zhuyangyang/Course/CMU10_414/homework/hw0/src/theta.csv";

    try {
        // Read CSV file into a vector of vectors of doubles
        std::vector<std::vector<double>> csvData = readCSV(csvFilePath);

        // std::cout<< "out func" << &csvData << std::endl;

        // Display the read data
        std::cout << "CSV data dimensions: " << csvData.size() << " rows x ";
        if (!csvData.empty()) {
            std::cout << csvData[0].size() << " columns" << std::endl;
        } else {
            std::cout << "0 columns" << std::endl;
        }

        // Access and process the data as needed
        // Example: Print the content of the CSV file
        // for (const auto& row : csvData) {
        //     for (double value : row) {
        //         std::cout << value << " ";
        //     }
        //     std::cout << std::endl;
        // }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
