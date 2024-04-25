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
# add_executable(test_readMNIST test/test_readMNIST.cpp src/readMNIST.cpp src/Tensor.cpp)
# add_executable(test_tensor test/test_tensor.cpp src/Tensor.cpp)
# add_executable(test_readCSV test/test_readCSV.cpp src/readCSV.cpp src/Tensor.cpp)
add_executable(forward_MNIST app/forward_MNIST.cpp src/readMNIST.cpp src/Tensor.cpp src/readCSV.cpp)
add_executable(forward_MNIST_conv app/forward_MNIST_conv.cpp src/readMNIST.cpp src/Tensor.cpp src/readCSV.cpp)
# add_executable(test_modules test/nn/test_modules.cpp src/nn/modules.cpp src/Tensor.cpp)

# Add include directories
include_directories(include)

# Find the zlib library
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    # Include zlib headers
    include_directories(${ZLIB_INCLUDE_DIRS})

    # Link against the zlib library
    # target_link_libraries(test_readMNIST ${ZLIB_LIBRARIES})
    target_link_libraries(forward_MNIST ${ZLIB_LIBRARIES})
    target_link_libraries(forward_MNIST_conv ${ZLIB_LIBRARIES})
else()
    message(FATAL_ERROR "Zlib library not found in the system, please install it first.")
endif()
