#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include "Device.hpp"
#include "Tensor.hpp"
#include "CUDA.hpp"
#include "CPU.hpp"
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

// Helper function to convert Python slices to std::vector<std::vector<int>>
std::vector<std::vector<int>> convert_slices(const py::list& py_slices, const std::vector<int>& tensor_shape) {
    std::vector<std::vector<int>> c_slices;
    
    for (size_t i = 0; i < py_slices.size(); ++i) {
        if (py::isinstance<py::slice>(py_slices[i])) {
            py::slice slice = py_slices[i].cast<py::slice>();
            
            ssize_t start, stop, step, length;
            
            std::cout << "slice: " << slice << std::endl;
            // Calculate start, stop, and step for this slice in the context of tensor_shape[i]
            // the first parameter of compute is the total size of the dimension
            if (!slice.compute(tensor_shape[i], &start, &stop, &step, &length)) {
                throw std::runtime_error("Invalid slice indices");
            }
            std::cout << "start: " << start << ", stop: " << stop << ", step: " << step << ", length: " << length << std::endl;

            // Add {start, stop, step} to c_slices for this dimension
            c_slices.push_back({static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step)});
        } else {
            throw std::invalid_argument("Each item must be a slice object.");
        }
    }

    if (c_slices.size() < tensor_shape.size()) {
        for (size_t i = c_slices.size(); i < tensor_shape.size(); ++i) {
            c_slices.push_back({}); // empyty vector means slice all
        }
    } else if (c_slices.size() > tensor_shape.size()) {
        throw std::invalid_argument("Too many slices provided.");
    }
    
    return c_slices;
}

std::vector<std::vector<int>> convert_slices(const py::tuple& py_slices, const std::vector<int>& tensor_shape) {
    std::vector<std::vector<int>> c_slices;

    for (size_t i = 0; i < py_slices.size(); ++i) {
        if (py::isinstance<py::slice>(py_slices[i])) {
            py::slice slice = py_slices[i].cast<py::slice>();
            
            ssize_t start, stop, step, length;
            
            // std::cout << "slice: " << slice << std::endl;
            // Calculate start, stop, and step for this slice in the context of tensor_shape[i]
            if (!slice.compute(tensor_shape[i], &start, &stop, &step, &length)) {
                throw std::runtime_error("Invalid slice indices");
            }
            // std::cout << "start: " << start << ", stop: " << stop << ", step: " << step << ", length: " << length << std::endl;

            // Add {start, stop, step} to c_slices for this dimension
            c_slices.push_back({static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step)});
        } else {
            throw std::invalid_argument("Each item must be a slice object.");
        }
    }

    // Handle cases where c_slices has fewer or more elements than tensor_shape
    if (c_slices.size() < tensor_shape.size()) {
        for (size_t i = c_slices.size(); i < tensor_shape.size(); ++i) {
            c_slices.push_back({}); // Empty vector implies slicing all elements in the dimension
        }
    } else if (c_slices.size() > tensor_shape.size()) {
        throw std::invalid_argument("Too many slices provided.");
    }

    return c_slices;
}

PYBIND11_MODULE(tensor_bindings, m) {

    // bind Tensor class(only bind float, will it cause error??)
    py::class_<Tensor<float>>(m, "Tensor_fp32")
        .def(py::init<const std::vector<int>&>())
        .def("shape", &Tensor<float>::shape)
        // .def("data", &Tensor<float>::data)

        .def("matmul", &Tensor<float>::matmul)

        // reduce functions
        .def("sum", &Tensor<float>::sum)
        .def("mean", &Tensor<float>::mean)
        .def("max", &Tensor<float>::max)
        .def("min", &Tensor<float>::min)
        .def("argmax", &Tensor<float>::argmax)
        .def("argmin", &Tensor<float>::argmin)

        // unary operations
        .def(-py::self)
        .def("sin", &Tensor<float>::sin)
        .def("cos", &Tensor<float>::cos)
        .def("exp", &Tensor<float>::exp)
        .def("log", &Tensor<float>::log)
        .def("abs", &Tensor<float>::abs)
        .def("tanh", &Tensor<float>::tanh)
        .def("silu", &Tensor<float>::silu)
        .def("sqrt", &Tensor<float>::sqrt)
        .def("rsqrt", &Tensor<float>::rsqrt)

        // binary operations see https://pybind11.readthedocs.io/en/stable/advanced/classes.html#operator-overloading
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def("pow", &Tensor<float>::pow)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        // set, get item
        .def("__getitem__", [](const Tensor<float>& self, const py::list& py_slices) {
            // Convert Python list of slices to std::vector<std::vector<int>>
            auto c_slices = convert_slices(py_slices, self.shape());
            return self.getItem(c_slices);
        })
        .def("__getitem__", [](const Tensor<float>& self, const py::tuple& py_slices) {
            // Convert Python tuple of slices to std::vector<std::vector<int>>
            auto c_slices = convert_slices(py_slices, self.shape());
            return self.getItem(c_slices);
        })
        .def("__setitem__", [](Tensor<float>& self, const py::tuple& indices, Tensor<float>& value) {
            auto c_slices = convert_slices(indices, self.shape());
            self.setItem(c_slices, value);
        })
        .def("__setitem__", [](Tensor<float>& self, const py::tuple& indices, float value) {
            auto c_slices = convert_slices(indices, self.shape());
            self.setItem(c_slices, value);
        })
        // .def("__setitem__", [](Tensor<float>& self, const std::vector<int>& indices, float value) {
        // })

        // accept slice in python, convert to vetor<vector<int>> pass to cpp method.
        .def("getItem", [](const Tensor<float>& self, const py::list& py_slices) {
            // Convert Python list of slices to std::vector<std::vector<int>>
            auto c_slices = convert_slices(py_slices, self.shape());
            return self.getItem(c_slices);
        })
        .def("getItem", [](const Tensor<float>& self, const py::tuple& py_slices) {
            // Convert Python tuple of slices to std::vector<std::vector<int>>
            auto c_slices = convert_slices(py_slices, self.shape());
            return self.getItem(c_slices);
        })
        // .def("setItem", &Tensor<float>::setItem, py::arg("slices"), py::arg("value"))

        .def("softmax", &Tensor<float>::softmax)
        .def("transpose", &Tensor<float>::transpose)

        // represent functions
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
        // float *data_ptr = new float[t.num_elements]; // error, not enough
        float *data_ptr = new float[t.device->size - t.offset_];
        if (t.device_type == "cpu") {
            // std::memcpy(data_ptr, t.device->getDataPtr() + t.offset_, t.num_elements * sizeof(float)); // error !!
            std::memcpy(data_ptr, t.device->getDataPtr() + t.offset_, (t.device->size - t.offset_) * sizeof(float)); // should copy all data after offset_, or may cause error due to stride ...
        } else if (t.device_type == "cuda") {
            CUDA_CHECK(cudaMemcpy(data_ptr, (float *)(t.device->getDataPtr()) + t.offset_, (t.device->size - t.offset_) * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            throw std::runtime_error("Unsupported device type: " + t.device_type);
        }

        return py::array_t<float>(t.shape_, numpy_strides, data_ptr);
    });

    // convert a numpy array to Tensor
    m.def("convert_to_tensor", [](py::array_t<float> a, const std::string& device_type) {
        // Get shape and strides from the numpy array and convert them to std::vector<int>
        std::vector<int> shape(a.ndim());
        std::vector<int> strides(a.ndim());

        for (size_t i = 0; i < a.ndim(); ++i) {
            shape[i] = static_cast<int>(a.shape(i));
            strides[i] = static_cast<int>(a.strides(i) / sizeof(float)); // Convert byte strides to element strides
        }

        float *data_ptr = nullptr;
        std::shared_ptr<Device<float>> device;

        if (device_type == "cpu") {
            data_ptr = new float[a.size()];
            std::memcpy(data_ptr, a.data(), a.size() * sizeof(float));
            device = std::shared_ptr<CPU<float>>(new CPU<float>(a.size(), data_ptr));
        } else if (device_type == "cuda") {
            CUDA_CHECK(cudaMalloc(&data_ptr, a.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(data_ptr, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
            device = std::shared_ptr<CUDA<float>>(new CUDA<float>(a.size(), data_ptr));
        } else {
            throw std::invalid_argument("Invalid device type. Supported types are 'cpu' and 'cuda'.");
        }

        // Construct and return the Tensor object
        // return Tensor<float>(std::move(shape), std::move(strides), 0, data_ptr, device_type);
        return Tensor<float>(std::move(shape), std::move(strides), 0, device, device_type);
    });
}
