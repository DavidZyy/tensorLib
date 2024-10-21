#include "Tensor.hpp"
#include "Transformer.hpp"

int main() {
    ModelArgs args;
    args.dim = 512;
    args.hidden_dim = 2048;
    args.n_layers = 12;
    args.n_heads = 1;
    args.vocab_size = 32000;
    args.max_batch_size = 1; // set 1 first
    args.max_seq_len = 1024;
    Transformer<float> transformer(args);
    Tensor<float> tokens({1, 8});
    for (int i = 0; i < 8; i++) {
        tokens.data_[i] = i;
    }
    auto a = transformer.forward(tokens, 0);
    std::cout << a << std::endl;
}
