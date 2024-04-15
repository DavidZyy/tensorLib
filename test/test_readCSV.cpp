#include "../include/readCSV.hpp"

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
