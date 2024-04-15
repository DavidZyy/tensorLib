#include "../include/readCSV.hpp"

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
