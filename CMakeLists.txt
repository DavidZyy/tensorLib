cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
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


# find cudatookit get error not find??
# error maybe caused by the conda env not set properly, check it. /raid/home/zhuyangyang/miniconda3/envs/llm/bin not in the PATH, maybe caused the error.
# Add CUDA support
find_package(CUDAToolkit REQUIRED)
    if (CUDAToolkit_FOUND)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        link_directories(${CUDAToolkit_LIBRARY_DIR})
    else()
        message(FATAL_ERROR "CUDAToolkit not found in the system, please install it first.")
    endif()

file(GLOB NN_SOURCES "tensorLib/src/nn/*.cpp")
file(GLOB OPS_SOURCES "tensorLib/src/ops/*.cpp")
file(GLOB DEVICE_SOURCES "tensorLib/src/device/*.cpp" "tensorLib/src/device/cpu/*.cpp" "tensorLib/src/device/cuda/*.cu")

# Add source files to create the library
set(ALL_SOURCES
    ${NN_SOURCES}
    ${OPS_SOURCES}
    ${DEVICE_SOURCES}
    tensorLib/src/Tensor.cpp
)

add_library(tensorLib STATIC ${ALL_SOURCES})

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
# https://forums.developer.nvidia.com/t/how-to-setup-a-cmake-project-with-cuda-debugging-support-in-vscode-ubuntu-20-04-nvidia-rtx-2080-max-q-driver-470-cuda-toolkit-11-5/213140
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tensorLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
    # set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

################################## tests ##########################################################

# Add tests
# add_executable(test_device tensorLib/test/test_device.cpp)
# target_link_libraries(test_device PRIVATE tensorLib)

add_executable(test_rms_norm 
    tensorLib/test/cuda_fuse_op/test_rms_norm.cpp
    # app/llm/src/Transformer.cpp 
    # tensorLib/src/nn/modules.cpp
)
target_link_libraries(test_rms_norm PRIVATE tensorLib)
target_include_directories(test_rms_norm PUBLIC 
    app/llm/include
    tensorLib/include
    tensorLib/include/device
    tensorLib/include/nn
)
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
# Set the PYBIND11_FINDPYTHON option to ON
# set(PYBIND11_FINDPYTHON ON)
# 
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
