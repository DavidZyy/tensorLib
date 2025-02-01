#include "Tokenizer.hpp"

const std::string prompt = "Once Upon a Time";
// const std::string tokenizer_path = "~/project/ML/mytorch/tokenizer.bin";
const std::string tokenizer_path = "./tokenizer.bin";

int main () {
    Tokenizer tokenizer(tokenizer_path, 32000);
    auto tokens = tokenizer.encode(prompt, true, false);
    
    // print tokens
    std::cout << "Tokens: ";
    for (auto t : tokens) {
        std::cout << t << " ";
    }
}

