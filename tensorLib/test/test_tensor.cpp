#include "Tensor.hpp"
#include <cstddef>
#include <vector>
#include "iostream"

int test_construct() {
    // Define a list of shapes for the tensors
    std::vector<std::vector<int>> shapes = {
        {},
        {1},
        {2},
        {2, 3},        // 2D tensor with shape (2, 3)
        {3, 4, 2},     // 3D tensor with shape (3, 4, 2)
        {2, 2, 2, 2}   // 4D tensor with shape (2, 2, 2, 2)
    };

    // Create and manipulate tensors with different shapes
    for (const auto& shape : shapes) {
        Tensor<int> tensor(shape);

        // Assign values to the tensor
        for (size_t i = 0; i < tensor.num_elements; ++i) {
            tensor.data_[i] = static_cast<int>(i+1);
        }

        // Print the tensor
        std::cout << "Tensor with shape ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) {
                std::cout << " x ";
            }
        }
        std::cout << ":" << std::endl;
        std::cout << tensor << std::endl;
        std::cout << std::endl;
    }

    return 0;
}


// Function to test the getData method of the Tensor class
int test_getData() {
    // Define a shape for the tensor (e.g., 2x3x4)
    // std::vector<int> shape = {2, 3, 4};
    std::vector<int> shape = {2, 3, 4};

    // Create a tensor with the specified shape
    Tensor<int> tensor(shape);

    // Fill the tensor with some test data (e.g., sequential numbers)
    int value = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                tensor.data_[i * shape[1] * shape[2] + j * shape[2] + k] = value++;
            }
        }
    }

    // Test the getData method by retrieving specific elements and checking the values
    // Expected values: tensor.getData({0, 0, 0}) = 0, tensor.getData({1, 2, 3}) = 23, etc.
    int expected_value = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                std::vector<int> indices = {i, j, k};
                int actual_value = tensor.getData(indices);
                if (actual_value != expected_value) {
                    std::cerr << "Error: Incorrect value retrieved at indices "
                              << "[" << i << "][" << j << "][" << k << "]. "
                              << "Expected: " << expected_value << ", Actual: " << actual_value << std::endl;
                    return 1; // Return failure
                }
                expected_value++;
            }
        }
    }

    std::cout << "getData method test passed!" << std::endl;
    return 0; // Return success
}

void test_setData() {
    // Create a Tensor object with a specific shape
    std::vector<int> shape = {5, 4, 3, 2};
    // std::vector<int> shape = {};
    Tensor<int> tensor(shape);

    // Set data at specific indices
    // tensor.setData({0, 0}, 1);
    // tensor.setData({0, 1}, 2);
    // tensor.setData({0, 2}, 3);
    // tensor.setData({1, 0}, 4);
    // tensor.setData({1, 1}, 5);
    // tensor.setData({1, 2}, 6);

    for(auto i=0; i<tensor.num_elements; i++)
        tensor.data_[i] = i;

    // Print the tensor to verify the data
    std::cout << "Tensor after setData:" << std::endl;
    std::cout << tensor << std::endl;
}

void test_parenthesis() {
    // Create a Tensor object with a specific shape
    std::vector<int> shape = {5, 4, 3, 2};
    Tensor<int> tensor(shape);

    int value = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for(int t = 0; t < shape[3]; ++t) {
                    tensor({i, j, k, t}) = value++;
                }
            }
        }
    }

    std::cout << "Tensor after setData:" << std::endl;
    std::cout << tensor << std::endl;
}

// void test_square_brackets() {
//     // Create a Tensor object with a specific shape
//     std::vector<int> shape = {5, 4, 3, 2};
//     Tensor<int> tensor(shape);
// 
//     int value = 0;
//     for (int i = 0; i < shape[0]; ++i) {
//         for (int j = 0; j < shape[1]; ++j) {
//             for (int k = 0; k < shape[2]; ++k) {
//                 for(int t = 0; t < shape[3]; ++t) {
//                     // tensor({i, j, k, t}) = value++;
//                     tensor[i][j][k][t] = value++;
//                 }
//             }
//         }
//     }
//     auto a = tensor[1][2][3][0];
// 
//     std::cout << "Tensor after setData:" << std::endl;
//     std::cout << tensor << std::endl;
//     // std::cout << tensor[1][2][3][0] << std::endl;
// }

int test_matmul() {
    std::vector<int> shape1 = {2, 3};
    std::vector<int> shape2 = {3, 2};
    Tensor<int> tensor1(shape1);
    Tensor<int> tensor2(shape2);

    // Fill the tensors with some test data
    for (int i = 0; i < shape1[0]; ++i) {
        for (int j = 0; j < shape1[1]; ++j) {
            tensor1({i, j}) = 1;
        }
    }

    for (int i = 0; i < shape2[0]; ++i) {
        for (int j = 0; j < shape2[1]; ++j) {
            tensor2({i, j}) = 1;
        }
    }

    Tensor<int> result = tensor1.matmul(tensor2);
    std::cout << "Result of matrix multiplication:" << std::endl;
    std::cout << result << std::endl;
    return 0;
}

Tensor<int> originTensor(const std::vector<int>& shape) {
    Tensor<int> tensor(shape);

    for(auto i=0; i<tensor.num_elements; i++)
        tensor.data_[i] = i;

    return tensor;
}


void test_view() {
    Tensor<int> a = originTensor({2, 3, 4, 5});
    auto b = a.view({2, 12, 5});
    auto c = b.view({5, 24});

    std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;
    std::cout << "c: " << std::endl << c << std::endl;

    /* test whether the data is shared, 
        whether the constructor make a deep copy or shadow copy. */
    std::cout << "address of a.data " << &a.data_[0] << std::endl;
    std::cout << "address of b.data " << &b.data_[0] << std::endl;
    std::cout << "address of c.data " << &c.data_[0] << std::endl;
}

void test_maximum() {
    Tensor<int> a = originTensor({2, 3});

    Tensor<int> temp({});

    temp.setData({}, 3);
    auto b = maximum(a, temp);

    std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;
}

void test_zeros() {
    auto a = zeros<int>({2, 3, 3,3});
    std::cout << "a: " << std::endl << a << std::endl;
}

void test_slice() {
    Tensor<int> a = originTensor({2, 3, 4, 5});
    // Tensor<int> b = a.slice(0, 1, 0);
    // Tensor<int> c = a.slice(0, 1, 1);
    // Tensor<int> d = a.slice(0, 2, 2);
    // Tensor<int> e = a.slice(0, 1, 3);
    // Tensor<int> f = a.slice(1, 2, 3);
    // Tensor<int> g = f.slice(1, 2, 2);
    // Tensor<int> h = g.slice(1, 2, 1);
    Tensor<int> i = a.slice(0, 1, 0).slice(0, 3, 2).slice(1, 4, 3);
    Tensor<int> j = a.slice(1, 2, 0).slice(1, 4, 2).slice(2, 5, 3);


    std::cout << "a: " << std::endl << a << std::endl;
    // std::cout << "b: " << std::endl << b << std::endl;
    // std::cout << "c: " << std::endl << c << std::endl;
    // std::cout << "e: " << std::endl << e << std::endl;
    // std::cout << "f: " << std::endl << f << std::endl;
    // std::cout << "g: " << std::endl << g << std::endl;
    // std::cout << "h: " << std::endl << h << std::endl;
    std::cout << "i: " << std::endl << i << std::endl;
    std::cout << "j: " << std::endl << j << std::endl;

    j.setData({0, 0, 0, 0}, 100);
    j.setData({0, 1, 1, 1}, 90);
    std::cout << "j: " << std::endl << j << std::endl;
}

/**
 * @brief only support thensor that have 4 dims
 */
int sum_up(Tensor<int> t) {
    assert(t.shape().size() == 4);
    int sum = 0;
    for (int i=0; i < t.shape()[0]; i++) {
        for (int j=0; j < t.shape()[1]; j++) {
            for (int k=0; k < t.shape()[2]; k++) {
                for (int l=0; l < t.shape()[3]; l++) {
                    sum += t.getData({i, j, k, l});
                }
            }
        }
    }

    return sum;
}

void test_sum() {
    Tensor<int> a = originTensor({2, 3, 4, 5});
    // Tensor<int> b = a.slice(0, 1, 0);
    // Tensor<int> c = a.slice(0, 1, 1);
    // Tensor<int> d = a.slice(0, 2, 2);
    // Tensor<int> e = a.slice(0, 1, 3);
    // Tensor<int> f = a.slice(1, 2, 3);
    // Tensor<int> g = f.slice(1, 2, 2);
    // Tensor<int> h = g.slice(1, 2, 1);
    Tensor<int> i = a.slice(0, 1, 0).slice(0, 3, 2).slice(1, 4, 3);
    Tensor<int> j = a.slice(1, 2, 0).slice(1, 4, 2).slice(2, 5, 3);


    assert(a.sum() == sum_up(a));
    std::cout << "a: " << std::endl << a << std::endl << "sum: " << a.sum() << " sum_up " << sum_up(a) << std::endl;
    // std::cout << "b: " << std::endl << b << std::endl;
    // std::cout << "c: " << std::endl << c << std::endl;
    // std::cout << "e: " << std::endl << e << std::endl;
    // std::cout << "f: " << std::endl << f << std::endl;
    // std::cout << "g: " << std::endl << g << std::endl;
    // std::cout << "h: " << std::endl << h << std::endl;
    assert(i.sum() == sum_up(i));
    std::cout << "i: " << std::endl << i << std::endl << "sum: " << i.sum() << " sum_up " << sum_up(i) << std::endl;
    assert(j.sum() == sum_up(j));
    std::cout << "j: " << std::endl << j << std::endl << "sum: " << j.sum() << " sum_up " << sum_up(j) << std::endl;

    j.setData({0, 0, 0, 0}, 100);
    j.setData({0, 1, 1, 1}, 90);
    assert(j.sum() == sum_up(j));
    std::cout << "j: " << std::endl << j << std::endl << "sum: " << j.sum() << " sum_up " << sum_up(j) << std::endl;
}

/**
 * @brief test elementwise mul, overload *.
 */
void test_elementwise_mul() {
    Tensor<int> a = originTensor({2, 3, 4, 5});

    Tensor<int> b = a.slice(0, 1, 0).slice(0, 1, 1);
    Tensor<int> c = a.slice(1, 2, 0).slice(2, 3, 1);

    // std::cout << "a: " << std::endl << a << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;
    std::cout << "c: " << std::endl << c << std::endl;
    std::cout << "b * c: " << std::endl << b*c << std::endl;
}

void test_select() {
    Tensor<int> a = originTensor({2, 3, 4, 5});
    // Tensor<int> b = a.slice(0, 1, 0);
    // Tensor<int> c = a.slice(0, 1, 1);
    // Tensor<int> d = a.slice(0, 2, 2);
    // Tensor<int> e = a.slice(0, 1, 3);
    // Tensor<int> f = a.slice(1, 2, 3);
    // Tensor<int> g = f.slice(1, 2, 2);
    // Tensor<int> h = g.slice(1, 2, 1);
    Tensor<int> i = a.select(3, 1);
    Tensor<int> j = i.select(0, 1);
    Tensor<int> k = j.select(1, 2);
    Tensor<int> t = i.slice(0, 2, 1);


    std::cout << "a: " << std::endl << a << std::endl;
    // std::cout << "b: " << std::endl << b << std::endl;
    // std::cout << "c: " << std::endl << c << std::endl;
    // std::cout << "e: " << std::endl << e << std::endl;
    // std::cout << "f: " << std::endl << f << std::endl;
    // std::cout << "g: " << std::endl << g << std::endl;
    // std::cout << "h: " << std::endl << h << std::endl;
    std::cout << "i: " << std::endl << i << std::endl;
    std::cout << "j: " << std::endl << j << std::endl;

    j.setData({0, 0}, 100);
    j.setData({0, 1}, 90);
    std::cout << "j: " << std::endl << j << std::endl;

    std::cout << "k: " << std::endl << k << std::endl;
    std::cout << "t: " << std::endl << t << std::endl;
}

/**
 * @brief test elementwise mul, overload *.
 */
void test_elementwise_mul_efficient() {
    Tensor<int> a = originTensor({200,500, 1, 1});
    Tensor<int> b = originTensor({200, 500,1 ,1});
    Tensor<int> c = a*b;

    std::cout << "the address of result: " << &c << std::endl;
    std::cout << "the data address of result: " << c.data() << std::endl;

//     Tensor<int> b = a.slice(0, 1, 0).slice(0, 1, 1);
//     Tensor<int> c = a.slice(1, 2, 0).slice(2, 3, 1);
// 
    // std::cout << "a: " << std::endl << a << std::endl;
//     std::cout << "b: " << std::endl << b << std::endl;
//     std::cout << "c: " << std::endl << c << std::endl;
//     std::cout << "b * c: " << std::endl << b*c << std::endl;
}

void test_set_get_item() {
    Tensor<int> a = originTensor({2, 3, 4, 5});
    std::cout << "a: " << std::endl << a << std::endl;

    std::vector<std::vector<int>> slices = {{0}, {1, 3}, {2, 4}, {1, 4}};

    auto b = a.getItem(slices);
    std::cout << "b: " << std::endl << b << std::endl;

    Tensor<int> value = originTensor(b.shape());
    std::cout << "value: " << std::endl << value << std::endl;

    a.setItem(slices, b);
    std::cout << "a: " << std::endl << a << std::endl;
}

void test_broadcast_to() {
    Tensor<int> a = originTensor({2, 3});

    std::cout << "a: " << std::endl << a << std::endl;

    auto b = a.view({2, 3, 1});
    std::cout << "b: " << std::endl << b << std::endl;

    auto c = a.view({2, 1, 3});
    std::cout << "c: " << std::endl << c << std::endl;

    b = b.broadcast_to({2,3,2});
    std::cout << "b: " << std::endl << b << std::endl;

    c = c.broadcast_to({2,2,3});
    std::cout << "c: " << std::endl << c << std::endl;
}

void test_max_or_sum() {
    Tensor<int> a = originTensor({2, 3, 4, 5});
    std::cout << "a: " << std::endl << a << std::endl;

    auto b = a.sum(0);
    std::cout << "b: " << std::endl << b << std::endl;

    auto c = a.sum(0, true);
    std::cout << "c: " << std::endl << c << std::endl;


    b = a.max(1);
    std::cout << "b: " << std::endl << b << std::endl;

    c = a.max(1, true);
    std::cout << "c: " << std::endl << c << std::endl;
}

int main() {
    // test_construct();
    // test_getData();
    // test_setData();
    // test_parenthesis();
    // test_square_brackets();
    // test_matmul();
    // test_view();
    // test_maximum();
    // test_zeros();
    // test_slice();
    // test_sum();
    // test_elementwise_mul();
    // test_select();
    // test_elementwise_mul_efficient();
    // test_set_get_item();
    // test_broadcast_to();
    test_max_or_sum();
}
