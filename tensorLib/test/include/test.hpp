#include "Tensor.hpp"

template <typename T> float get_max_abs_difference(const Tensor<T>& a, const Tensor<T>& b);
template <typename T> bool check_equal(const Tensor<T>& a, const Tensor<T>& b);

template <typename T> void check_equal_and_max_diff(Tensor<T>& a, Tensor<T>& b);