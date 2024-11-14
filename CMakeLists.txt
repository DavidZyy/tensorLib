cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
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

# Add CUDA support
find_package(CUDA REQUIRED)
    if (CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    else()
        message(FATAL_ERROR "CUDA not found in the system, please install it first.")
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

# Enable position-independent code
set_target_properties(tensorLib PROPERTIES POSITION_INDEPENDENT_CODE ON)  # for tensor_bindings.so

# vscode Debug cuda kernel, see https://stackoverflow.com/questions/67888279/debugging-cuda-kernels-with-vs-code
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tensorLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

################################## tests ##########################################################

# Add tests
add_executable(test_device tensorLib/test/test_device.cpp)
target_link_libraries(test_device PRIVATE tensorLib)

################################## llama2 ##########################################################
add_executable(llama2
    app/llm/src/main.cpp 
    app/llm/src/llama2.cpp 
    app/llm/src/Transformer.cpp 
    app/llm/src/Tokenizer.cpp 
)

target_include_directories(llama2 PUBLIC 
    app/llm/include
)

target_link_libraries(llama2 PRIVATE tensorLib)

################################## tensor_bindings module ##########################################################

# # pybind11
# find_package(pybind11 REQUIRED)
# include_directories(${pybind11_INCLUDE_DIRS})
# 
# find_package(Python3 REQUIRED)
# include_directories(${Python3_INCLUDE_DIRS})
# 
# # Add your source files
# add_library(tensor_bindings MODULE tensorLib/test/tensor_bindings.cpp)
# 
# # Set the output name without the 'lib' prefix
# set_target_properties(tensor_bindings
#     PROPERTIES
#     PREFIX ""
# )
# 
# # Link with pybind11
# target_link_libraries(tensor_bindings PRIVATE pybind11::module tensorLib)
# # pybind11_strip(tensor_bindings)
