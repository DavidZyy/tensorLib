cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
################################# compile tensorLib library #############################################

# Adjust based on your target GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

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
    tensorLib/src/Log.cpp
)

add_library(tensorLib STATIC ${ALL_SOURCES})

target_include_directories(tensorLib PUBLIC 
    tensorLib/include
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

target_link_libraries(tensorLib PRIVATE ${CUDA_LIBRARIES} cublas)
################################## correctness tests ##########################################################

# Add tests
# add_executable(test_device tensorLib/test/test_device.cpp)
# target_link_libraries(test_device PRIVATE tensorLib)

# add_executable(test_rms_norm 
#     tensorLib/test/cuda_fuse_op/test_rms_norm.cpp
# )
# target_link_libraries(test_rms_norm PRIVATE tensorLib)
# target_include_directories(test_rms_norm PUBLIC 
#     model/llama2_7b/include
#     tensorLib/include
# )

# add_executable(sgemm_benchmark
#     tensorLib/test/cuda_benchmark/sgemm_benchmark.cpp
# )
# target_link_libraries(sgemm_benchmark PRIVATE tensorLib)

# add_executable(test_fp16
#     tensorLib/test/fp16/test_fp16.cu
#     model/llama2_7b/src/Transformer.cpp
# )
# target_link_libraries(test_fp16 PRIVATE tensorLib)
# target_include_directories(test_fp16 PUBLIC 
#     model/llama2_7b/include
#     tensorLib/include
# )
################################## cuda kernel benchmark ##########################################################
add_executable(hgemv_benchmark
    tensorLib/test/cuda_benchmark/hgemv_benchmark.cu
)

target_link_libraries(hgemv_benchmark PRIVATE tensorLib)
################################## llama2 ##########################################################
add_executable(llama2
    model/llama2_7b/src/main.cpp 
    model/llama2_7b/src/llama2.cpp 
    model/llama2_7b/src/Transformer.cpp 
    model/llama2_7b/src/Tokenizer.cpp 
)

target_include_directories(llama2 PUBLIC 
    model/llama2_7b/include
)

target_link_libraries(llama2 PRIVATE tensorLib)

################################## tensor_bindings module ##########################################################
# # Set the PYBIND11_FINDPYTHON option to ON
# set(PYBIND11_FINDPYTHON ON)
# 
# # pybind11
# find_package(pybind11 REQUIRED)
# include_directories(${pybind11_INCLUDE_DIRS})
# 
# find_package(Python3 REQUIRED)
# include_directories(${Python3_INCLUDE_DIRS})
# 
# # Add source files
# add_library(tensor_bindings MODULE tensorLib/test/python/tensor_bindings.cpp)
# 
# # Set the output name without the 'lib' prefix
# set_target_properties(tensor_bindings
#     PROPERTIES
#     PREFIX ""
# )
# 
# # Link with pybind11
# target_link_libraries(tensor_bindings PRIVATE pybind11::module tensorLib)
# pybind11_strip(tensor_bindings)
