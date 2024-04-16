cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set debug build type
set(CMAKE_BUILD_TYPE Debug)

# Add debug flags to the compiler options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Add executable target
# add_executable(a a.cpp)
# add_executable(b b.cpp)
add_executable(test_readMNIST test/test_readMNIST.cpp src/readMNIST.cpp src/Tensor.cpp)
add_executable(test_tensor test/test_tensor.cpp src/Tensor.cpp)
add_executable(test_readCSV test/test_readCSV.cpp src/readCSV.cpp src/Tensor.cpp)

# Add include directories
include_directories(include)

# Find the zlib library
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    # Include zlib headers
    include_directories(${ZLIB_INCLUDE_DIRS})

    # Link against the zlib library
    target_link_libraries(test_readMNIST ${ZLIB_LIBRARIES})
else()
    message(FATAL_ERROR "Zlib library not found.")
endif()
