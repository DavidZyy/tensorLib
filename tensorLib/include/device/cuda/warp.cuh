


// only lane 0 holds the result value, if all lanes need the result, should broadcast
// to all lanes
// template<typename T>
// __inline__ __device__ T warpReduceSum(T val) {
//     for (int offset = 16; offset > 0; offset /= 2) {
//         val += __shfl_down_sync(0xFFFFFFFF, val, offset);
//     }
//     return val;
// }
// 
// template<typename T>
// __inline__ __device__ T warpReduceMax(T val) {
//     for (int offset = 16; offset > 0; offset /= 2) {
//         T other = __shfl_down_sync(0xFFFFFFFF, val, offset);
//         val = (val > other) ? val : other; // Generic max
//     }
//     return val;
// }

// all lanes hold the result value.
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = (val > other) ? val : other; // Generic max
    }
    return val;
}