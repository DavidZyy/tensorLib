// #include "gtest/gtest.h"
// #include "../include/Tensor.hpp"  // Include the header file containing the Tensor class
// 
// // Test fixture for the Tensor class
// class TensorTest : public ::testing::Test {
// protected:
//     // Create a helper method to generate a sample Tensor instance
//     template <typename T>
//     static Tensor<T> createSampleTensor(std::initializer_list<int> shape) {
//         std::vector<T> sample_data;  // Initialize with sample data for your tests
//         return Tensor<T>(shape, sample_data);
//     }
// 
//     // Test cases can access this member
//     Tensor<float> test_tensor;
// };
// 
// // Test case for the getData() method
// TYPED_TEST_SUITE_P(TensorTest);

// Test valid indices
// TYPED_TEST_P(TensorTest, TestValidIndices) {
//     std::vector<int> indices = {1, 2, 3};  // Adjust indices for your test tensor shape
//     EXPECT_NO_THROW(this->test_tensor.getData(indices));
//     // Add additional assertions to verify the returned value against the expected one
// }
// 
// // Test invalid indices size
// TYPED_TEST_P(TensorTest, TestInvalidIndicesSize) {
//     std::vector<int> indices = {1, 2, 3, 4};  // Adjust indices size to exceed the test tensor's dimensions
//     EXPECT_THROW(this->test_tensor.getData(indices), std::invalid_argument);
// }
// 
// // Test index out of range
// TYPED_TEST_P(TensorTest, TestIndexOutOfRange) {
//     std::vector<int> indices = {-1, 2, 3};  // Adjust indices to contain at least one out-of-range value
//     EXPECT_THROW(this->test_tensor.getData(indices), std::out_of_range);
// }
// 
// // Register the test cases in the TensorTest typed test suite
// REGISTER_TYPED_TEST_SUITE_P(TensorTest,
//                              TestValidIndices,
//                              TestInvalidIndicesSize,
//                              TestIndexOutOfRange);
// 
// // Instantiate the typed tests for the desired data types
// using DataTypes = ::testing::Types<float, double, int>;  // Adjust the list of data types as needed
// INSTANTIATE_TYPED_TEST_SUITE_P(My, TensorTest, DataTypes);


// #include "gtest/gtest.h"
// #include "../include/Tensor.hpp"  // Include the header file containing the Tensor class
// 
// // Test fixture for the Tensor class
// class TensorTest : public ::testing::Test {
// protected:
//     // Create a helper method to generate a sample Tensor instance
//     template <typename T>
//     static Tensor<T> createSampleTensor(std::initializer_list<int> shape) {
//         std::vector<T> sample_data;  // Initialize with sample data for your tests
//         return Tensor<T>(shape, sample_data);
//     }
// 
//     // Test cases can access this member
//     Tensor<float> test_tensor;
// };
// 
// // Test case for the getData() method
// TEST(TensorTest, TestValidIndices) {
//     std::vector<int> indices = {1, 2, 3};  // Adjust indices for your test tensor shape
//     EXPECT_NO_THROW(test_tensor.getData(indices));
//     // Add additional assertions to verify the returned value against the expected one
// }
// 
// TYPED_TEST_P(TensorTest, TestValidIndices) {
//     std::vector<int> indices = {1, 2, 3};  // Adjust indices for your test tensor shape
//     EXPECT_NO_THROW(this->test_tensor.getData(indices));
//     // Add additional assertions to verify the returned value against the expected one
// }
// 
// // Test invalid indices size
// TEST(TensorTest, TestInvalidIndicesSize) {
//     std::vector<int> indices = {1, 2, 3, 4};  // Adjust indices size to exceed the test tensor's dimensions
//     EXPECT_THROW(test_tensor.getData(indices), std::invalid_argument);
// }
// 
// // Test index out of range
// TEST(TensorTest, TestIndexOutOfRange) {
//     std::vector<int> indices = {-1, 2, 3};  // Adjust indices to contain at least one out-of-range value
//     EXPECT_THROW(test_tensor.getData(indices), std::out_of_range);
// }