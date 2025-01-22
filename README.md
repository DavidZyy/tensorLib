目录说明:

```cpp
.
├── app // application codes
├── CMakeLists.txt
├── doc // document
├── include
├── README.md
├── src // library codes
└── test // test codes
```


构建代码：
```
mkdir build
cd build
cmake ..
make
```
# Debug
Uncomment `set(CMAKE_BUILD_TYPE Debug)` in `CmakeLists.txt`.

# Test
## python test
use vscode test tool.

# Acknowledgement
This project inspired by following projects:

CMU10_414
llama2.c
llm.c
llama.cpp
simpleTensor
