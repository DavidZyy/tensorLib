# cmake_minimum_required(VERSION 3.10)
# project(MyProject LANGUAGES CXX CUDA)
# # project(MyProject)
# 
# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# 
# # Set debug build type
# # set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
# 
# # Add debug flags to the compiler options
# # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
# 
# # Add OpenMP support
# find_package(OpenMP REQUIRED)
# 
# if(OpenMP_CXX_FOUND)
#     # Append OpenMP flags to the compiler
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()
# 
# # Add executable target
# # add_executable(test_readMNIST tensorLib/test/test_readMNIST.cpp tensorLib/src/readMNIST.cpp tensorLib/src/Tensor.cpp)
# # add_executable(test_tensor tensorLib/test/test_tensor.cpp tensorLib/src/Tensor.cpp)
# # add_executable(test_readCSV tensorLib/test/test_readCSV.cpp tensorLib/src/readCSV.cpp tensorLib/src/Tensor.cpp)
# # add_executable(test_modules tensorLib/test/nn/test_modules.cpp tensorLib/src/nn/modules.cpp tensorLib/src/Tensor.cpp)
# # add_executable(test_shared_ptr tensorLib/test/test_shared_ptr.cpp)
# 
# # add_executable(forward_MNIST app/forward_MNIST.cpp tensorLib/src/readMNIST.cpp tensorLib/src/Tensor.cpp tensorLib/src/readCSV.cpp)
# # add_executable(forward_MNIST_conv app/forward_MNIST_conv.cpp tensorLib/src/readMNIST.cpp tensorLib/src/Tensor.cpp tensorLib/src/readCSV.cpp)
# # add_executable(forward_MNIST_quantize app/forward_MNIST_quantize.cpp tensorLib/src/readMNIST.cpp tensorLib/src/Tensor.cpp tensorLib/src/readCSV.cpp)
# 
# # add_executable(test_tokenizer app/llm/test/test_tokenizer.cpp app/llm/src/Tokenizer.cpp)
# 
# # add_executable(test_transformer 
# #     app/llm/test/test_transformer.cpp 
# #     app/llm/src/Transformer.cpp 
# #     tensorLib/src/Tensor.cpp
# #     tensorLib/src/nn/modules.cpp
# #     )
# 
# add_executable(main
#     app/llm/src/main.cpp 
#     app/llm/src/llama2.cpp 
#     app/llm/src/Transformer.cpp 
#     app/llm/src/Tokenizer.cpp 
#     tensorLib/src/Tensor.cpp
#     tensorLib/src/nn/modules.cpp
#     )
# 
# # Add include directories
# include_directories(tensorLib/include)
# include_directories(app/llm/include)
# 
# # Find the zlib library
# find_package(ZLIB REQUIRED)
# if (ZLIB_FOUND)
#     # Include zlib headers
#     include_directories(${ZLIB_INCLUDE_DIRS})
# 
#     # Link against the zlib library
#     # target_link_libraries(test_readMNIST ${ZLIB_LIBRARIES})
#     # target_link_libraries(forward_MNIST ${ZLIB_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
#     # target_link_libraries(forward_MNIST_conv ${ZLIB_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
#     # target_link_libraries(forward_MNIST_quantize ${ZLIB_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
# else()
#     message(FATAL_ERROR "Zlib library not found in the system, please install it first.")
# endif()
# 
# # pybind11
# # find_package(pybind11 REQUIRED)
# # include_directories(${pybind11_INCLUDE_DIRS})
# # 
# # find_package(Python3 REQUIRED)
# # include_directories(${Python3_INCLUDE_DIRS})
# # 
# # # Add your source files
# # add_library(tensor_bindings MODULE tensorLib/test/tensor_bindings.cpp tensorLib/src/Tensor.cpp)
# # 
# # # Set the output name without the 'lib' prefix
# # set_target_properties(tensor_bindings
# #     PROPERTIES
# #     PREFIX ""
# # )
# # 
# # # Link with pybind11
# # target_link_libraries(tensor_bindings PRIVATE pybind11::module)
# # pybind11_strip(tensor_bindings)
# 
# 
# # CUDA configuration
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
# link_directories(${CUDA_LIBRARY_DIRS})
# 
# # Optionally, specify CUDA architecture
# set(CMAKE_CUDA_ARCHITECTURES 75)
# 
# add_compile_options(-x cuda tensorLib/include/device)
# link_libraries(cudart) # Links CUDA runtime library


cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

################################# compile tensorLib library #############################################

# Adjust based on your target GPU
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add OpenMP support
find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        # Append OpenMP flags to the compiler
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message(FATAL_ERROR "OpenMP not found in the system, please install it first.")
    endif()

# Add source files to create the library
set(SRC
    tensorLib/src/device/CPU.cpp
    tensorLib/src/device/CUDA.cu
    tensorLib/src/device/Device.cpp
    tensorLib/src/nn/modules.cpp
    tensorLib/src/Tensor.cpp
)

add_library(tensorLib STATIC ${SRC})

target_include_directories(tensorLib PUBLIC 
    tensorLib/include
    tensorLib/include/device
    tensorLib/include/nn
)

set_target_properties(tensorLib
    PROPERTIES
    PREFIX ""
)

################################## linke library to application ##########################################################

