#include "Tokenizer.hpp"
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

Tokenizer::Tokenizer(const std::string& tokenizer_path, int vocab_size) : vocab_size(vocab_size) {
    this->vocab.resize(vocab_size);
    this->vocab_scores.resize(vocab_size);

    for (int i = 0; i < 256; i++) {
        this->byte_pieces[i * 2] = static_cast<unsigned char>(i);
        this->byte_pieces[i * 2 + 1] = '\0';
    }

    // Read the tokenizer file
    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) {
        std::cerr << "couldn't load " << tokenizer_path << std::endl;
        throw std::runtime_error("File loading error");
    }


    // Read max_token_length
    file.read(reinterpret_cast<char*>(&this->max_token_length), sizeof(int));
    if (!file) {
        std::cerr << "failed to read max_token_length" << std::endl;
        throw std::runtime_error("File reading error");
    }

    // Read vocab_scores and vocab strings
    for (int i = 0; i < vocab_size; ++i) {
        // Read vocab_scores
        file.read(reinterpret_cast<char*>(&this->vocab_scores[i]), sizeof(float));
        if (!file) {
            std::cerr << "failed to read vocab_scores" << std::endl;
            throw std::runtime_error("File reading error");
        }

        // Read length of the string
        int len;
        file.read(reinterpret_cast<char*>(&len), sizeof(int));
        if (!file) {
            std::cerr << "failed to read string length" << std::endl;
            throw std::runtime_error("File reading error");
        }

        // Read vocab string
        std::string str(len, '\0'); // Allocate memory for string
        file.read(&str[0], len);    // Read string data
        if (!file) {
            std::cerr << "failed to read vocab string" << std::endl;
            throw std::runtime_error("File reading error");
        }
        this->vocab[i] = std::move(str); // Store the string in vocab
    }

    // Close the file
    file.close();
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) {
    std::vector<int> tokens;

    // if (text.empty()) {
    //     std::cerr << "cannot encode empty text" << std::endl;
    //     std::exit(EXIT_FAILURE);
    // }
    
    // set the sorted_vocab for find "token ID" according to the "token string".
    if (this->sorted_vocab.empty()) {
        this->sorted_vocab.reserve(this->vocab_size);
        for (int i = 0; i < this->vocab_size; ++i) {
            this->sorted_vocab.emplace_back(TokenIndex(this->vocab[i], i));
        }

        std::sort(this->sorted_vocab.begin(), this->sorted_vocab.end(), compare_tokens);
    }

    std::string str_buffer;
    str_buffer.reserve(max_token_length * 2 + 1 + 2);
    
    if (bos) tokens.push_back(1);

    if (!text.empty()) {
        int dummy_prefix = str_lookup(" ", sorted_vocab, vocab_size);
        tokens.push_back(dummy_prefix);
    }

    // a UTF-8 string may comprised with 1 or 2 or 3 or 4 bytes, if a byte is part of utf-8 string, 
    // not the first byte of the string, the first 2 bits of the byte is "10",
    // so the following codes find the whole utf-8 string's bytes and append them to the buffer,
    // and then use the string to find the token id.
    size_t str_len = 0; // the byte counts of the current utf-8 string
    for (auto it = text.begin(); it != text.end(); ++it) {
        // this is the first byte of a utf-8 string, so we reset the counter.
        if ((*it & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer += *it;
        str_len++;
        
        // the next byte is still a part of this utf-8 string, so we continue appending bytes to str_buffer.
        if ((*(it + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok, now we have a whole utf-8 string, we can find its token id.
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

        if (id != -1) {
            tokens.push_back(id);
        } else {
            for (int i=0; i < str_len; i++) {
                tokens.push_back(static_cast<unsigned char>(str_buffer[i]) + 3);
            }
        }

        str_buffer.clear();
        str_len = 0;
    }


    // now merge the best consecutive pair each iteration, according the scores in vacab_scores.
    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < tokens.size() - 1; ++i) {
            std::string combined = this->vocab[tokens[i]] + this->vocab[tokens[i + 1]];
            int id = str_lookup(combined, sorted_vocab, vocab_size);

            if (id != -1 && this->vocab_scores[id] > best_score) {
                best_score = this->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        // can not find any consecutive pair to merge, so we stop.
        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    if (eos) tokens.push_back(2);

    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) {
    std::string& piece = this->vocab[token];

    if (prev_token == 1 && piece[0] == ' ') {
        piece = piece.substr(1);
    }

    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        piece = std::string(reinterpret_cast<const char*>(&this->byte_pieces[byte_val * 2]));
    }

    return piece;
}
