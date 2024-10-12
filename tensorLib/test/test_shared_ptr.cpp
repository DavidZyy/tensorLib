
// #include <iostream>
// #include <memory>
// using namespace std;
// 
// void deleter(int* ptr) {
//     cout << "Memory freed!" << endl;
//     delete[] ptr;
// }
// 
// int main () {
//     int * p = new int[10];
//     // assign to p:
//     for (int i=0; i<10; i++) p[i]=i;
// 
//     // shared_ptr<int[]> p1(new int[10]);
//     shared_ptr<int[]> p1(p);
// 
//     auto p2 = p1;
// 
//     // cout p1
//     for (int i=0; i<10; i++) cout << p1[i] << " ";
//     cout << endl;
// 
//     // cout p2
//     for (int i=0; i<10; i++) cout << p2[i] << " ";
//     cout << endl;
// }

// #include <iostream>
// #include <memory>
// using namespace std;
// 
// void deleter(int* ptr) {
//     cout << "Memory freed!" << endl;
//     delete[] ptr;
// }
// 
// int main () {
//     {
//         // Create shared_ptr with custom deleter
//         shared_ptr<int[]> p1(new int[10], deleter);
//         
//         // assign to p1:
//         for (int i=0; i<10; i++) p1[i] = i;
// 
// 
// 
//         {
//             auto p2 = p1; // p2 shares ownership of p1's array
//             cout << "users: " << p1.use_count() << endl;
//         }
//         cout << "users: " << p1.use_count() << endl;
// 
//         // Memory is not freed yet because both p1 and p2 are still in scope
//         cout << "Scope not ended" << endl;
//     }
// 
//     cout << "Scope ended" << endl;
//     // Now both p1 and p2 go out of scope, and the memory is freed only once.
// }


#include <cstddef>
#include <iostream>
#include <memory>
#include <atomic>

using namespace std;

// Global variable to track allocated memory
static atomic<size_t> totalAllocatedMemory(0);
static atomic<size_t> totalFreedMemory(0);

class Deleter {
public:
    void operator()(int* ptr) const {
        // 实现删除逻辑
        totalFreedMemory += sizeof(int) * allocatedSize;
        delete[] ptr;
        cout << "Memory freed!" << endl;
    }
    Deleter(size_t size) : allocatedSize(size) {}

private:
    size_t allocatedSize;
};


class MyClass {
public:
    MyClass(size_t size) : size_(size) {
        std::shared_ptr<int[]> temp(new int[size], Deleter(size));
        // data_ = static_cast<shared_ptr<int[]>>(new int[10], deleter);
        data_ = temp;
        totalAllocatedMemory += sizeof(int) * size;
    }
    MyClass(const MyClass& other) {
        data_ = other.data_; // shared_ptr handles copying and reference count
    }
    MyClass(MyClass&& other) noexcept {
        data_ = std::move(other.data_); // Transfers ownership, no reference count increment
    }

    ~MyClass() {}

    static size_t GetTotalAllocatedMemory() {
        return totalAllocatedMemory.load();
    }

    static size_t GetTotlFreedMemory() {
        return totalFreedMemory.load();
    }

// private:
    shared_ptr<int[]> data_;
private:
    size_t size_;
};

// int main () {
//     {
// 
//         auto i1 = MyClass(10);
//         cout << "i1 users: " << i1.data_.use_count() << endl;
//     {
//         // auto i2 = i1;
//         auto i2 = std::move(i1);
//         // auto i2(std::move(i1));
//         // auto i2(i1);
//         cout << "i1 users: " << i1.data_.use_count() << endl;
//         cout << "i2 users: " << i2.data_.use_count() << endl;
//     }
//         cout << "i1 users: " << i1.data_.use_count() << endl;
// 
//         cout << "Scope not ended" << endl;
//     }
// 
//     cout << "Scope ended" << endl;
// }

int main() {
    cout << "Initial memory usage: " << MyClass::GetTotalAllocatedMemory() << " bytes"<< endl;
    cout << "Initial memory usage: " << MyClass::GetTotlFreedMemory() << " bytes"<< endl;

    {
        MyClass i1(10);
        cout << "Memory usage after creating i1: " << MyClass::GetTotalAllocatedMemory() << " bytes" << endl;
        cout << "Memory usage after creating i1: " << MyClass::GetTotlFreedMemory() << " bytes" << endl;
        {
            auto i2 = i1;
            // auto i2 = std::move(i1);
            cout << "Memory usage after moving i1 to i2: " << MyClass::GetTotalAllocatedMemory() << " bytes" << endl;
            cout << "Memory usage after moving i1 to i2: " << MyClass::GetTotlFreedMemory() << " bytes" << endl;
        }
        cout << "Memory usage after i2 goes out of scope: " << MyClass::GetTotalAllocatedMemory() << " bytes" << endl;
        cout << "Memory usage after i2 goes out of scope: " << MyClass::GetTotlFreedMemory() << " bytes" << endl;
    }
    cout << "Final memory usage: " << MyClass::GetTotalAllocatedMemory() << " bytes" << endl;
    cout << "Final memory usage: " << MyClass::GetTotlFreedMemory() << " bytes" << endl;
}
