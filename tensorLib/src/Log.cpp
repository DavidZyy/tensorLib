#include "Log.hpp"

// Singleton pattern for the Logger instance
Logger& Logger::instance() {
    static Logger instance("log.txt");
    return instance;
}

Logger::Logger(const std::string& filename) {
    // logFile.open(filename, std::ios::app); // Open in append mode
    logFile.open(filename, std::ios::trunc); // Open in truncate mode to clear file
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file!" << std::endl;
    }
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

void Logger::writeLog(LogLevel level, const std::string& message) {
     std::lock_guard<std::mutex> lock(logMutex); // Ensure thread safety
    if (logFile.is_open()) {
        logFile << getCurrentTime() << " [" << logLevelToString(level) << "] " << message << std::endl;
    }
}

std::string Logger::getCurrentTime() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_info = std::localtime(&now);
    std::ostringstream oss;
    oss << (tm_info->tm_year + 1900) << "-"
        << (tm_info->tm_mon + 1) << "-"
        << tm_info->tm_mday << " "
        << tm_info->tm_hour << ":"
        << tm_info->tm_min << ":"
        << tm_info->tm_sec;
    return oss.str();
}

std::string Logger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}
