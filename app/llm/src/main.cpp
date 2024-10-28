#include "Transformer.hpp"
#include "llama2.hpp"
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <new>
#include <string>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h> // Add this line at the top of your file

// const std::string prompt = "One";
const std::string prompt = "In your daily life";
const std::string tokenizer_path = "./tokenizer.bin";
const std::string checkpoint = "./stories15M.bin";

// from llama2.c project
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

void memory_map_weights(Llama2<float>& generator, float* ptr, int shared_weight) {
    ModelArgs p = generator.model.params;

    int head_size = p.dim / p.n_heads;
    size_t n_layers = p.n_layers;
    generator.model.tok_embeddings.weight.data_ = std::shared_ptr<float[]>(ptr);
    ptr += p.vocab_size * p.dim;
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention_norm.weight.data_ = std::shared_ptr<float[]>(ptr);
        // std::cout << "attention_norm " << l << std::endl;
        // std::cout << layer->attention_norm.weight << std::endl;
        ptr += p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wq.weight.data_ = std::shared_ptr<float[]>(ptr);
        // std::cout << "attention.wq " << l << std::endl;
        // if (l == 0) {
        //     auto a = layer->attention.wq.weight.slice(0, 1, 0);
        //     std::cout << a << std::endl;
        // }
        ptr += p.dim * (p.n_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wk.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.dim * (p.n_kv_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wv.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.dim * (p.n_kv_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wo.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.dim * (p.n_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->ffn_norm.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->feed_forward.w1.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.hidden_dim * p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->feed_forward.w2.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.dim * p.hidden_dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->feed_forward.w3.weight.data_ = std::shared_ptr<float[]>(ptr);
        ptr += p.hidden_dim * p.dim;
    }
    generator.model.norm.weight.data_ = std::shared_ptr<float[]>(ptr);
    ptr += p.dim;
    // ptr += p.seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    // ptr += p.seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    ptr += 256 * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += 256 * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    generator.model.output.weight.data_ = shared_weight ? generator.model.tok_embeddings.weight.data_ : std::shared_ptr<float[]>(ptr);
    // generator.model.norm
}

void read_weights(Llama2<float>& generator,  int shared_weight) {
    ModelArgs p = generator.model.params;
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str()); exit(EXIT_FAILURE); }

    int head_size = p.dim / p.n_heads;
    size_t n_layers = p.n_layers;

    size_t cur_off = sizeof(Config);
    // fseek(file, cur_off, SEEK_SET);

    float *ptr = new float[p.vocab_size * p.dim];
    fseek(file, cur_off, SEEK_SET);
    fread(ptr, sizeof(float), p.vocab_size * p.dim, file);
    // cur_off += p.vocab_size * p.dim * sizeof(float);
    generator.model.tok_embeddings.weight.data_ = std::shared_ptr<float[]>(ptr);
    // std::cout << "tok_embeddings.weight " << std::endl;
    // std::cout << generator.model.tok_embeddings.weight << std::endl;

    for (size_t l = 0; l < n_layers; l++) {
        float* ptr = new float[p.dim]; //allocate 
        // fseek(file, cur_off, SEEK_SET);
        fread(ptr, sizeof(float), p.dim, file);
        // cur_off += p.dim * sizeof(float);

        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention_norm.weight.data_ = std::shared_ptr<float[]>(ptr);
        // std::cout << "attention_norm " << l << std::endl;
        // std::cout << layer->attention_norm.weight << std::endl;
        // ptr += p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wq.weight.data_ = std::shared_ptr<float[]>(ptr);
        // std::cout << "attention.wq " << l << std::endl;
        // std::cout << layer->attention.wq.weight << std::endl;
        // ptr += p.dim * (p.n_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wk.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.dim * (p.n_kv_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wv.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.dim * (p.n_kv_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->attention.wo.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.dim * (p.n_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float* ptr = new float[p.dim]; //allocate 
        fread(ptr, sizeof(float), p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->ffn_norm.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.hidden_dim * p.dim];
        fread(ptr, sizeof(float), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->feed_forward.w1.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.hidden_dim * p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.hidden_dim * p.dim];
        fread(ptr, sizeof(float), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->feed_forward.w2.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.dim * p.hidden_dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.hidden_dim * p.dim];
        fread(ptr, sizeof(float), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        layer->feed_forward.w3.weight.data_ = std::shared_ptr<float[]>(ptr);
        // ptr += p.hidden_dim * p.dim;
    }
    ptr = new float[p.dim]; //allocate 
    fread(ptr, sizeof(float), p.dim, file);
    generator.model.norm.weight.data_ = std::shared_ptr<float[]>(ptr);
    // std::cout << "norm " << std::endl;
    // std::cout << generator.model.norm.weight << std::endl;
    // ptr += p.dim;
    // ptr += p.seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    // ptr += p.seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    // ptr += 256 * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    // ptr += 256 * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    assert(shared_weight == 1);
    generator.model.output.weight.data_ = shared_weight ? generator.model.tok_embeddings.weight.data_ : std::shared_ptr<float[]>(ptr);
    // generator.model.norm
    fclose(file);
}

/**
 * read args and parameters from checkpoint file.
 */
Llama2<float> read_checkpoint() {
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str()); exit(EXIT_FAILURE); }
    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weight = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = std::abs(config.vocab_size);
    // fseek(file, 0, SEEK_END);
    // *file_size = ftell(file);
    // size_t fs = ftell(file);
    fclose(file);

    ModelArgs args;
    args.dim = config.dim;
    args.hidden_dim = config.hidden_dim;
    args.n_layers = config.n_layers;
    args.n_heads = config.n_heads;
    args.n_kv_heads = config.n_kv_heads;
    args.vocab_size = config.vocab_size;
    args.max_seq_len = config.seq_len;
    args.max_batch_size = 1;

    auto generator = Llama2<float>(tokenizer_path, args);

//     *fd = open(checkpoint.c_str(), O_RDONLY);
//     if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
//     *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
//     if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
// 
//     float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    // memory_map_weights(generator, weights_ptr, shared_weight);
    read_weights(generator, shared_weight);
    return generator;
}

int main() {
    ModelArgs args;
    // args.dim = 256;
    args.dim = 16;
    // args.hidden_dim = 1024;
    args.hidden_dim = 32;
    // args.n_layers = 12;
    args.n_layers = 1;
    args.n_heads = 1;
    args.vocab_size = 32000;
    args.max_batch_size = 1; // set 1 first
    args.max_seq_len = 1024;

    // auto generator = Llama2<float>(tokenizer_path, args);
    // float **data = new float*;
    // int* fd = new int;
    // size_t* file_size = new size_t;
    auto generator = read_checkpoint();
    generator.text_completion(prompt);

    // munmap data and close fd here!
    // if (data != MAP_FAILED) { munmap(*data, *file_size); }
    // if (*fd != -1) { close(*fd); }
    // delete data;
    // delete fd;
    // delete file_size;
}
