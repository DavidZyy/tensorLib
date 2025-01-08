#include "device/CUDA.hpp"
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

// const std::string prompt = "Once";
const std::string prompt = "";
// const std::string prompt = "Once upon a time";
const std::string tokenizer_path = "../tokenizer.bin";
// const std::string checkpoint = "../stories15M.bin";
// const std::string checkpoint = "../stories42M.bin";
// const std::string checkpoint = "../stories110M.bin";
// const std::string checkpoint = "/raid/home/zhuyangyang/LLAMA/llama2.c/llama2_7b.bin";

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

void read_weights(char* checkpoint, Llama2<float>& generator,  int shared_weight) {
    std::string device_type = generator.model.device_type;

    ModelArgs p = generator.model.params;
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    int head_size = p.dim / p.n_heads;
    size_t n_layers = p.n_layers;

    size_t cur_off = sizeof(Config);
    // fseek(file, cur_off, SEEK_SET);

    float *ptr = new float[p.vocab_size * p.dim];
    fseek(file, cur_off, SEEK_SET);
    fread(ptr, sizeof(float), p.vocab_size * p.dim, file);
    // cur_off += p.vocab_size * p.dim * sizeof(float);
    // generator.model.tok_embeddings.weight.data_ = std::shared_ptr<float[]>(ptr);
    if (device_type == "cpu") {
        std::memcpy(generator.model.tok_embeddings.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(float));
    } else if (device_type == "cuda") {
        CUDA_CHECK(cudaMemcpy(generator.model.tok_embeddings.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(float), cudaMemcpyHostToDevice));
    }
    delete [] ptr;
    // std::cout << "tok_embeddings.weight " << std::endl;
    // std::cout << generator.model.tok_embeddings.weight << std::endl;

    for (size_t l = 0; l < n_layers; l++) {
        float* ptr = new float[p.dim]; //allocate 
        // fseek(file, cur_off, SEEK_SET);
        fread(ptr, sizeof(float), p.dim, file);
        // cur_off += p.dim * sizeof(float);

        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->attention_norm.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->attention_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // std::cout << "attention_norm " << l << std::endl;
        // std::cout << layer->attention_norm.weight << std::endl;
        // ptr += p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->attention.wq.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wq.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wq.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // std::cout << "attention.wq " << l << std::endl;
        // std::cout << layer->attention.wq.weight << std::endl;
        // ptr += p.dim * (p.n_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->attention.wk.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wk.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wk.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.dim * (p.n_kv_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->attention.wv.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wv.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wv.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.dim * (p.n_kv_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(float), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->attention.wo.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wo.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wo.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.dim * (p.n_heads * head_size);
    }
    for (size_t l = 0; l < n_layers; l++) {
        float* ptr = new float[p.dim]; //allocate 
        fread(ptr, sizeof(float), p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->ffn_norm.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->ffn_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->ffn_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.hidden_dim * p.dim];
        fread(ptr, sizeof(float), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->feed_forward.w1.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->feed_forward.w1.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->feed_forward.w1.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim* sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.hidden_dim * p.dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.hidden_dim * p.dim];
        fread(ptr, sizeof(float), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->feed_forward.w2.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->feed_forward.w2.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->feed_forward.w2.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim* sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.dim * p.hidden_dim;
    }
    for (size_t l = 0; l < n_layers; l++) {
        float *ptr = new float[p.hidden_dim * p.dim];
        fread(ptr, sizeof(float), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<float>>(generator.model.layers[l]);
        // layer->feed_forward.w3.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(layer->feed_forward.w3.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->feed_forward.w3.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim* sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
        // ptr += p.hidden_dim * p.dim;
    }
    ptr = new float[p.dim]; //allocate 
    fread(ptr, sizeof(float), p.dim, file);
    // generator.model.norm.weight.data_ = std::shared_ptr<float[]>(ptr);
    if (device_type == "cpu") {
        std::memcpy(generator.model.norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(float));
    } else if (device_type == "cuda") {
        CUDA_CHECK(cudaMemcpy(generator.model.norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(float), cudaMemcpyHostToDevice));
    }
    delete [] ptr;

    if (shared_weight) {
        // generator.model.output.weight.data_ = generator.model.tok_embeddings.weight.data_;
        generator.model.output.weight.device = generator.model.tok_embeddings.weight.device;
        // if (device_type == "cpu") {
        //     std::memcpy(generator.model.output.weight.device->getDataPtr(), generator.model.tok_embeddings.weight.device->getDataPtr(), p.vocab_size * p.dim * sizeof(float));
        // } else if (device_type == "cuda") {
        //     CUDA_CHECK(cudaMemcpy(generator.model.output.weight.device->getDataPtr(), generator.model.tok_embeddings.weight.device->getDataPtr(), p.vocab_size * p.dim * sizeof(float), cudaMemcpyHostToDevice));
        // }
    } else {
        // fseek(file, p.max_seq_len * head_size, SEEK_CUR); // rope
        ptr = new float[p.max_seq_len * head_size];
        fread(ptr, sizeof(float), p.max_seq_len* head_size, file);
        delete [] ptr;

        ptr = new float[p.vocab_size * p.dim];
        fread(ptr, sizeof(float), p.vocab_size * p.dim, file);
        // generator.model.output.weight.data_ = std::shared_ptr<float[]>(ptr);
        if (device_type == "cpu") {
            std::memcpy(generator.model.output.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(float));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(generator.model.output.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(float), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }

    // generator.model.norm
    fclose(file);
}

/**
 * read args and parameters from checkpoint file.
 */
Llama2<float> read_checkpoint(char * checkpoint) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weight = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = std::abs(config.vocab_size);
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

    // auto generator = Llama2<float>(tokenizer_path, args, "cpu");
    auto generator = Llama2<float>(tokenizer_path, args, "cuda");

    read_weights(checkpoint, generator, shared_weight);
    return generator;
}

int main(int argc, char *argv[]) {
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

    auto generator = read_checkpoint(argv[1]);
    generator.text_completion(prompt);
}
