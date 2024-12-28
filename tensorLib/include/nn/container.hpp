#pragma once
#include "nn/modules.hpp"

namespace nn {

// Define the ModuleList class
template <typename dtype> class ModuleList : public Module<dtype> {
public:
  // Default constructor
  ModuleList() = default;
  ModuleList(std::string device_type);

  // Append a new module to the list
  void append(std::shared_ptr<Module<dtype>> module) {
    modules_.push_back(module);
  }

  // Access a module by index
  std::shared_ptr<Module<dtype>> operator[](int index) const {
    if (index < 0 || index >= modules_.size()) {
      throw std::out_of_range("ModuleList index out of range.");
    }
    return modules_[index];
  }

  // Get the size of the ModuleList
  int size() const { return modules_.size(); }

  // Forward pass through all modules in the list
  Tensor<dtype> forward(const Tensor<dtype> &input) const {
    Tensor<dtype> current_input = input;

    // Sequentially pass through all modules in the list
    for (const auto &module : modules_) {
      current_input = module->forward(current_input);
    }

    return current_input;
  }

private:
  // Vector to hold the list of shared pointers to modules
  std::vector<std::shared_ptr<Module<dtype>>> modules_;
};

}