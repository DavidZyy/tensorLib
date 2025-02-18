
/**
    NOTE: this file mainly check the correctness of fp16 in cuda, in cpu not been checked. use fp32 module to test fp16 module.
 */
#include "Tensor.hpp"
#include "device/CUDA.hpp"
#include "nn/rmsNorm.hpp"
#include <iostream>
#include <chrono>
#include "Transformer.hpp"
#include "test.hpp"

#define MEASURE_TIME(code_block) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration<double>(end - start); \
        std::cout << "Execution time: " << duration.count() << " seconds" << std::endl; \
    } while (0)

/************************************************************************************************************************************************************/

void test_matmul() {
    int bs = 1;
    // int m = 128;
    int m = 1;
    int n = 4096;
    int k = 4096;

    Tensor<float> a = randn<float>({bs, m, k}, "cuda");
    Tensor<float> b = randn<float>({bs, k, n}, "cuda");
    Tensor<float> c;
    MEASURE_TIME({
        c = a.matmul(b); 
    });

    // std::cout << "a: " << std::endl << a << std::endl;
    // std::cout << "b: " << std::endl << b << std::endl;
    // std::cout << "c: " << std::endl << c << std::endl;

    Tensor<half> a_fp16(a);
    Tensor<half> b_fp16(b);
    Tensor<half> _c_fp16;
    MEASURE_TIME({
        _c_fp16 = a_fp16.matmul(b_fp16);
    });

    // std::cout << "c: " << std::endl << c << std::endl;
    // std::cout << "c_fp16: " << std::endl << c_fp16 << std::endl;
    // std::cout << "_c_fp16: " << std::endl << _c_fp16 << std::endl;

    Tensor<half> c_fp16(c);
    check_equal_and_max_diff(c_fp16, _c_fp16);
}

/************************************************************************************************************************************************************/

void test_rms() {
    int n = 1;
    int dim = 4096;
    // int dim = 10;
    float eps = 1e-5;

    // fp32
    nn::RMSNorm<float> rms_fp32 = nn::RMSNorm<float>(dim, eps, "cuda");
    Tensor<float> x = randn<float>({n, dim}, "cuda");
    Tensor<float> y;
    // Tensor<float> y1;
    MEASURE_TIME({
        y = rms_fp32.forward_fused_cuda(x);
        // y = rms_fp32.forward_plain(x);
        // y1 = rms_fp32.forward_plain(x);
    });

    std::cout << "x: " << std::endl << x << std::endl;
    std::cout << "weight:" << std::endl << rms_fp32.weight << std::endl;
    std::cout << "y: " << std::endl << y << std::endl;
    // std::cout << "y1: " << std::endl << y1 << std::endl;

    // fp16
    nn::RMSNorm<half> rms_fp16 = nn::RMSNorm<half>(dim, eps, "cuda");
    // set weight
    Tensor<half> weight_fp16(rms_fp32.weight);
    rms_fp16.weight = weight_fp16;

    Tensor<half> x_fp16(x);
    Tensor<half> y_fp16;
    MEASURE_TIME({
        y_fp16 = rms_fp16.forward_fused_cuda(x_fp16);
        // y_fp16 = rms_fp16.forward_plain(x_fp16);
    });

    std::cout << "x_fp16: " << std::endl << x_fp16 << std::endl;
    std::cout << "weight_fp16:" << std::endl << rms_fp16.weight << std::endl;
    std::cout << "y_fp16: " << std::endl << y_fp16 << std::endl;

    // check equal and get max_diff
    Tensor<half> _y_fp16(y);

    check_equal_and_max_diff(y_fp16, _y_fp16);
}

/************************************************************************************************************************************************************/

// void test_attention() {
//     ModelArgs args;
//     args.n_heads = 128;
//     args.dim = 4096;
//     args.max_batch_size = 1;
//     args.max_seq_len = 2048;
// 
//     // fp32
//     Attention<float> attention_fp32 = Attention<float>(args, "cuda");
//     Tensor<float> x = randn<float>({1, args.dim}, "cuda");
//     Tensor<float> y;
//     MEASURE_TIME({
//         y = attention_fp32.forward(x);
//     });
// 
//     // fp16
// 
// }

/************************************************************************************************************************************************************/

void test_ffn() {
    int dim = 1024;
    int hidden_dim = 1024;
    // int dim = 10;
    // int hidden_dim = 100;

    // fp32
    FeedForward<float> ffn_fp32 = FeedForward<float>(dim, hidden_dim, "cuda");
    // set weight
    Tensor<float> w1 = randn<float>({hidden_dim, dim}, "cuda");
    Tensor<float> w2 = randn<float>({dim, hidden_dim}, "cuda");
    Tensor<float> w3 = randn<float>({hidden_dim, dim}, "cuda");
    ffn_fp32.w1.weight = w1;
    ffn_fp32.w2.weight = w2;
    ffn_fp32.w3.weight = w3;

    Tensor<float> x = randn<float>({1, dim}, "cuda");
    Tensor<float> y;
    MEASURE_TIME({
        y = ffn_fp32.forward(x);
    });

    std::cout << "x: " << std::endl << x << std::endl;
    std::cout << "y: " << std::endl << y << std::endl;

    // fp16
    FeedForward<half> ffn_fp16 = FeedForward<half>(dim, hidden_dim, "cuda");
    // set weight
    Tensor<half> w1_fp16(w1);
    Tensor<half> w2_fp16(w2);
    Tensor<half> w3_fp16(w3);
    ffn_fp16.w1.weight = w1_fp16;
    ffn_fp16.w2.weight = w2_fp16;
    ffn_fp16.w3.weight = w3_fp16;

    Tensor<half> x_fp16(x);
    Tensor<half> y_fp16;
    MEASURE_TIME({
        y_fp16 = ffn_fp16.forward(x_fp16);
    });

    std::cout << "x_fp16: " << std::endl << x_fp16 << std::endl;
    std::cout << "y_fp16: " << std::endl << y_fp16 << std::endl;

    Tensor<half> _y_fp16(y);

    check_equal_and_max_diff(y_fp16, _y_fp16);
}

/************************************************************************************************************************************************************/

void test_argmax() {
    int m = 10;
    int n = 32000;

    Tensor<float> x = randn<float>({m, n}, "cuda");
    Tensor<int> y;
    MEASURE_TIME({
        y = x.argmax(1);
    });

    // std::cout << "x: " << std::endl << x << std::endl;
    // std::cout << "y: " << std::endl << y << std::endl;

    Tensor<half> x_fp16(x);
    Tensor<int> y_fp16;
    MEASURE_TIME({
        y_fp16 = x_fp16.argmax(1);
    });

    // std::cout << "x_fp16: " << std::endl << x_fp16 << std::endl;
    // std::cout << "y_fp16: " << std::endl << y_fp16 << std::endl;

    std::cout << "y: " << std::endl << y << std::endl;
    std::cout << "y_fp16: " << std::endl << y_fp16 << std::endl;

    // check_equal_and_max_diff(y_fp16, y);
}

/************************************************************************************************************************************************************/

int main() {
    // test_matmul();
    // test_rms();
    // test_ffn();
    test_argmax();
    return 0;
}
