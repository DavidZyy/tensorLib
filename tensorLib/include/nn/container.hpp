#pragma once
#include "nn/modules.hpp"

namespace nn {

// Define the ModuleList class
template <typename dtype> class ModuleList : public Module<dtype> {
public:
  // Default constructor
    ModuleList() = default;
    ModuleList(std::string device_type);

    void append(std::shared_ptr<Module<dtype>> module);
    std::shared_ptr<Module<dtype>> operator[](int index) const;
    int size() const;
    Tensor<dtype> forward(const Tensor<dtype> &input) const;

private:
  // Vector to hold the list of shared pointers to modules
  std::vector<std::shared_ptr<Module<dtype>>> modules_;
};

}
