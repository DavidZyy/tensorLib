/**
 * @file Tokenizer.hpp
 * @author Yangyang Zhu (yangyangzhu12@qq.com)
 * @version 0.1
 * @date 2024-10-14
 * 
 * @copyright Copyright (c) 2024
 * This is a tokenizer, modified from llama2.c (https://github.com/karpathy/llama2.c)
 */

#pragma once

#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <algorithm>

class TokenIndex {
public:
    std::string str;  // Using std::string for dynamic memory management
    int id;

    TokenIndex(const std::string& s = "", int i = 0) : str(s), id(i) {}
};

class Tokenizer {
public:
    // build tokenizer
    Tokenizer(const std::string& tokenizer_file, int vocab_size);

    std::vector<int> encode(const std::string& text, bool bos, bool eos); // return tokens

    std::string decode(int prev_token, int token); // return decoded tokens (string)

// public:
private:
    std::vector<std::string> vocab;          // Replacing char** with std::vector<std::string>
    std::vector<float> vocab_scores;         // Replacing float* with std::vector<float>
    std::vector<TokenIndex> sorted_vocab;    // Replacing TokenIndex* with std::vector<TokenIndex>
    int vocab_size;
    unsigned int max_token_length;
    std::array<unsigned char, 512> byte_pieces; // Using std::array for fixed-size array

    static bool compare_tokens(const TokenIndex &a, const TokenIndex &b) {
        // return strcmp(a.str.c_str(), b.str.c_str()) < 0;
        return a.str < b.str;
    }

    // find the token with the given string in the sorted vocabulary
    int str_lookup(const std::string& str, const std::vector<TokenIndex>& sorted_vocab, int vocab_size) {
        TokenIndex key{str, -1};

        auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), key, compare_tokens);

        if (it != sorted_vocab.end() && it->str == str) return it->id;

        return -1;
    }

};

