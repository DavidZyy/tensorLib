#include "nn/container.hpp"

namespace nn  {

template class ModuleList<float>;
template class ModuleList<int>;

template <typename dtype>
ModuleList<dtype>::ModuleList(std::string device_type) : Module<dtype>(device_type) {}

// Append a new module to the list
template <typename dtype>
void ModuleList<dtype>::append(std::shared_ptr<Module<dtype>> module) {
    modules_.push_back(module);
}

// Access a module by index
template <typename dtype>
std::shared_ptr<Module<dtype>> ModuleList<dtype>::operator[](int index) const {
    if (index < 0 || index >= modules_.size()) {
        throw std::out_of_range("ModuleList index out of range.");
    }
    return modules_[index];
}

// Get the size of the ModuleList
template <typename dtype>
int ModuleList<dtype>::size() const { return modules_.size(); }

// Forward pass through all modules in the list
template <typename dtype>
Tensor<dtype> ModuleList<dtype>::forward(const Tensor<dtype> &input) const {
    Tensor<dtype> current_input = input;

    // Sequentially pass through all modules in the list
    for (const auto &module : modules_) {
        current_input = module->forward(current_input);
    }

    return current_input;
}


} // namespace nn
