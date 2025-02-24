#include "device/cuda/CUDA.cuh"
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
#include <chrono>

// const std::string prompt = "Once";
// const std::string prompt = "";
const std::string prompt = "Once upon a time";
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

template <typename dtype>
void read_weights(char* checkpoint, Llama2<dtype>& generator,  int shared_weight) {
    std::string device_type = generator.model.device_type;

    ModelArgs p = generator.model.params;
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    int head_size = p.dim / p.n_heads;
    size_t n_layers = p.n_layers;

    size_t cur_off = sizeof(Config);

    dtype *ptr = new dtype[p.vocab_size * p.dim];
    fseek(file, cur_off, SEEK_SET);
    fread(ptr, sizeof(dtype), p.vocab_size * p.dim, file);

    if (device_type == "cpu") {
        std::memcpy(generator.model.tok_embeddings.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(dtype));
    } else if (device_type == "cuda") {
        CUDA_CHECK(cudaMemcpy(generator.model.tok_embeddings.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(dtype), cudaMemcpyHostToDevice));
    }
    delete [] ptr;

    for (size_t l = 0; l < n_layers; l++) {
        dtype* ptr = new dtype[p.dim]; //allocate 
        fread(ptr, sizeof(dtype), p.dim, file);

        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->attention_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(dtype), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wq.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wq.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(dtype), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wk.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wk.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(dtype), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wv.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wv.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    // wo
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.dim * (p.n_heads * head_size)];
        fread(ptr, sizeof(dtype), p.dim * (p.n_heads * head_size), file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->attention.wo.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->attention.wo.weight.device->getDataPtr(), ptr, p.dim * (p.n_heads * head_size) * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    // ffn_norm
    for (size_t l = 0; l < n_layers; l++) {
        dtype* ptr = new dtype[p.dim]; //allocate 
        fread(ptr, sizeof(dtype), p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->ffn_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->ffn_norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    // feed_forward.w1
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.hidden_dim * p.dim];
        fread(ptr, sizeof(dtype), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->feed_forward.w1.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->feed_forward.w1.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim* sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.hidden_dim * p.dim];
        fread(ptr, sizeof(dtype), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->feed_forward.w2.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->feed_forward.w2.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim* sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    for (size_t l = 0; l < n_layers; l++) {
        dtype *ptr = new dtype[p.hidden_dim * p.dim];
        fread(ptr, sizeof(dtype), p.hidden_dim * p.dim, file);
        auto layer = std::dynamic_pointer_cast<TransformerBlock<dtype>>(generator.model.layers[l]);
        if (device_type == "cpu") {
            std::memcpy(layer->feed_forward.w3.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(layer->feed_forward.w3.weight.device->getDataPtr(), ptr, p.hidden_dim * p.dim* sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }
    ptr = new dtype[p.dim]; //allocate 
    fread(ptr, sizeof(dtype), p.dim, file);
    if (device_type == "cpu") {
        std::memcpy(generator.model.norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(dtype));
    } else if (device_type == "cuda") {
        CUDA_CHECK(cudaMemcpy(generator.model.norm.weight.device->getDataPtr(), ptr, p.dim * sizeof(dtype), cudaMemcpyHostToDevice));
    }
    delete [] ptr;

    if (shared_weight) {
        generator.model.output.weight.device = generator.model.tok_embeddings.weight.device;
    } else {
        ptr = new dtype[p.max_seq_len * head_size];
        fread(ptr, sizeof(dtype), p.max_seq_len* head_size, file);
        delete [] ptr;

        ptr = new dtype[p.vocab_size * p.dim];
        fread(ptr, sizeof(dtype), p.vocab_size * p.dim, file);
        if (device_type == "cpu") {
            std::memcpy(generator.model.output.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(dtype));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(generator.model.output.weight.device->getDataPtr(), ptr, p.vocab_size * p.dim * sizeof(dtype), cudaMemcpyHostToDevice));
        }
        delete [] ptr;
    }

    fclose(file);
}

/**
 * read args and parameters from checkpoint file.
 */
template<typename dtype>
Llama2<dtype> read_checkpoint(char * checkpoint) {
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

    // auto generator = Llama2<dtype>(tokenizer_path, args, "cpu");
    auto generator = Llama2<dtype>(tokenizer_path, args, "cuda");

    std::cout <<"Loading model weights..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    read_weights<dtype>(checkpoint, generator, shared_weight);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;
    std::cout << "Load time: " << elapsed_time.count() << " seconds" << std::endl;

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

    // auto generator = read_checkpoint<float>(argv[1]);
    auto generator = read_checkpoint<half>(argv[1]);
    // generator.text_completion(prompt);
    generator.chat();
}
