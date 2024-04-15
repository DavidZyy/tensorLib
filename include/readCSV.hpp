#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

// Function to read CSV file into a vector of vectors of doubles
std::vector<std::vector<double>> readCSV(const std::string& filename);
    