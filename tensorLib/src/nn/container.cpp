#include "nn/container.hpp"

namespace nn  {

template class ModuleList<float>;
template class ModuleList<int>;

template <typename dtype>
ModuleList<dtype>::ModuleList(std::string device_type) : Module<dtype>(device_type) {}

} // namespace nn
