#include "device/cpu/CPU.hpp"

template class CPU<int8_t>;
template class CPU<half>;
template class CPU<float>;
template class CPU<int>;


template<typename dtype>
void softmaxImpl(dtype* output, const dtype* input, size_t rows, size_t cols) {
    # pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {

        dtype max = input[i * cols];
        for (size_t j = 1; j < cols; j++) {
            if (input[i * cols + j] > max) {
                max = input[i * cols + j];
            }
        }
        dtype sum = 0;
        for (size_t j = 0; j < cols; j++) {
            output[i * cols + j] = exp(input[i * cols + j] - max);
            sum += output[i * cols + j];
        }

        for (size_t j = 0; j < cols; j++) {
            output[i * cols + j] /= sum;
        }

    }
}


template<typename dtype>
void CPU<dtype>::softmax(dtype* output, size_t rows, size_t cols) const {
    softmaxImpl(output, this->data_, rows, cols);
}
