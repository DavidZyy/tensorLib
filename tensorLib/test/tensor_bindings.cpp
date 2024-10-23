#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Tensor.hpp"
#include <sstream>

// Function to convert vector to string
// std::string vector_to_string(const std::vector<int>& vec) {
//     std::ostringstream oss;
//     oss << "[";
//     for (size_t i = 0; i < vec.size(); ++i) {
//         oss << vec[i];
//         if (i != vec.size() - 1) {
//             oss << ", ";
//         }
//     }
//     oss << "]";
//     return oss.str();
// }

namespace py = pybind11;

PYBIND11_MODULE(tensor_bindings, m) {

    // bind Tensor class
    py::class_<Tensor<float>>(m, "Tensor")
        .def(py::init<const std::vector<int>&>())
        .def("shape", &Tensor<float>::shape)
        // .def("data", &Tensor<float>::data)
        .def("matmul", &Tensor<float>::matmul)
        .def("transpose", &Tensor<float>::transpose)
        .def("argmax", &Tensor<float>::argmax)
        .def("sum", &Tensor<float>::sum)
        .def("mean", &Tensor<float>::mean)
        .def("max", &Tensor<float>::max)
        .def("softmax", &Tensor<float>::softmax)

        .def("__repr__", [](const Tensor<float>& t) { // used for debugging
            std::ostringstream oss;
            oss << "Tensor(shape=(";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                oss << t.shape()[i];
                if (i < t.shape().size() - 1) {
                    oss << ", ";
                }
            }
            oss << "), data=" << t << ")";
            return oss.str();
        })
        .def("__str__", [](const Tensor<float>& t) { // used for printing
            std::ostringstream oss;
            oss << t;
            return oss.str();
        });
        
    // convert a Tensor to numpy array
    m.def("convert_to_numpy", [](const Tensor<float>& t) {
        std::vector<int> numpy_strides = t.stride_;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
               [](int& c) { return c * sizeof(float); });
        // assert(t.offset_ == 0); // not yet handle this case now.
        float* data_ptr = t.data_.get();
        return py::array_t<float>(t.shape_, numpy_strides, data_ptr + t.offset_);
    });

    // convert a numpy array to Tensor
    m.def("convert_to_tensor", [](py::array_t<float> a) {
        // Get shape and strides from the numpy array and convert them to std::vector<int>
        std::vector<int> shape(a.ndim());
        std::vector<int> strides(a.ndim());

        for (size_t i = 0; i < a.ndim(); ++i) {
            shape[i] = static_cast<int>(a.shape(i));
            strides[i] = static_cast<int>(a.strides(i) / sizeof(float)); // Convert byte strides to element strides
        }

        // Wrap the numpy array data into a shared_ptr with a custom deleter (to avoid double-free)
        auto data_ptr = std::shared_ptr<float[]>(a.mutable_data(), [](float* p) {
            // Numpy owns the memory, so no need to delete p
        });

        // Construct and return the Tensor object
        return Tensor<float>(std::move(shape), std::move(strides), 0, data_ptr);
    });
}
