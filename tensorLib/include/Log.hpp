#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>
#include <mutex>
#include <vector>

// Define log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// Logger class definition
class Logger {
public:
    static Logger& instance();

    template <typename... Args>
    void log(LogLevel level, Args&&... args) {
        std::ostringstream oss;
        // (oss << ... << args); // C++17 fold expression (or use a loop for C++11)
        (printToStream(oss, std::forward<Args>(args)), ...); // C++17 fold expression
        writeLog(level, oss.str());
    }

private:
    Logger(const std::string& filename = "log.txt");
    ~Logger();

    Logger(const Logger&) = delete;       // Prevent copying
    Logger& operator=(const Logger&) = delete; // Prevent assignment

    std::ofstream logFile;
    std::mutex logMutex; // Ensure thread safety

    void writeLog(LogLevel level, const std::string& message);
    std::string getCurrentTime();
    std::string logLevelToString(LogLevel level);

    // Helper function for printing different types to stream
    template <typename T>
    void printToStream(std::ostringstream& oss, const T& value) {
        oss << value;
    }

    // Specialization for std::vector<T>
    template <typename T>
    void printToStream(std::ostringstream& oss, const std::vector<T>& vec) {
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            oss << vec[i];
            if (i < vec.size() - 1) oss << ", ";
        }
        oss << "]";
    }
};

// Macros for convenience
#define LOG_DEBUG(...)   Logger::instance().log(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)    Logger::instance().log(LogLevel::INFO, __VA_ARGS__)
#define LOG_WARNING(...) Logger::instance().log(LogLevel::WARNING, __VA_ARGS__)
#define LOG_ERROR(...)   Logger::instance().log(LogLevel::ERROR, __VA_ARGS__)
