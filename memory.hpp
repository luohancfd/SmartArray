/**
 * @file memory.hpp
 * @author Han Luo (han.luo@gmail.com)
 * @brief Utilities to create continuous memory for c type data
 * @version 0.1
 * @date 2020-03-20
 *
 * @copyright Copyright (c) 2020
 * @license   MIT
 *
 */

#pragma once
#include <algorithm>
#include <vector>
#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <array>
#include <cstdlib>
#include <type_traits>

namespace Memory {

template <typename DT> inline DT *createC(int m) { return new DT[m]; }

template <typename DT> inline void freeC(DT *array) {
  if (array)
    delete[] array;
}

template <typename DT> inline DT **createC(int m, int n) {
  DT **array = new DT *[m];
  *array = new DT[m * n];
  for (int i = 1; i < m; ++i) {
    array[i] = array[i - 1] + n;
  }
  return array;
}

template <typename DT> inline void freeC(DT **array) {
  if (array) {
    if (array[0])
      delete[] array[0];
    delete[] array;
  }
}

template <typename DT> inline DT ***createC(int m, int n, int k) {
  DT ***array = new DT **[m];
  *array = new DT *[m * n];
  **array = new DT[m * n * k];

  for (int i = 1; i < m; ++i) {
    array[i] = array[i - 1] + n;
    array[i][0] = array[i - 1][0] + n * k;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 1; j < n; j++) {
      array[i][j] = array[i][j - 1] + k;
    }
  }
  return array;
}

template <typename DT> inline void freeC(DT ***array) {
  if (array) {
    if (array[0]) {
      if (array[0][0]) {
        delete[] array[0][0];
      }
      delete[] array[0];
    }
    delete[] array;
  }
}

template <typename T, int dim> struct pointer_n {
  static_assert(dim < 4, "dimension must be smaller than 4");
  static_assert(dim > 0, "dimension must be greater than 0");
  typedef T type;
};

template <typename T> struct pointer_n<T, 1> { typedef T *type; };

template <typename T> struct pointer_n<T, 2> { typedef T **type; };

template <typename T> struct pointer_n<T, 3> { typedef T ***type; };

template <typename T, std::size_t dim>
using pointer_n_t = typename pointer_n<T, dim>::type;

template <typename DT, int dim>
pointer_n_t<DT, dim> createC(std::array<int, dim> shape) {
  if constexpr (dim == 1) {
    return createC<DT>(shape[0]);
  }
  if constexpr (dim == 2) {
    return createC<DT>(shape[0], shape[1]);
  }
  if constexpr (dim == 3) {
    return createC<DT>(shape[0], shape[1], shape[2]);
  }
  return nullptr;
}

/**
 * @brief Get the pointer to the first data
 *
 * @tparam T
 * @tparam dim
 * @param b
 * @return T*
 */
template <typename T, int dim> T *ptrC(pointer_n_t<T, dim> b) {
  if constexpr (dim == 1) {
    return b;
  }
  if constexpr (dim == 2) {
    return *b;
  }
  if constexpr (dim == 3) {
    return **b;
  }
  return nullptr;
}

template <typename T, int dim>
void copyC(pointer_n_t<T, dim> a, pointer_n_t<T, dim> b, int size) {
  T *start = ptrC<T, dim>(a);
  T *end = ptrC<T, dim>(a) + size;
  std::copy(start, end, ptrC<T, dim>(b));
}

template <typename T, int dim>
void fillC(T a, pointer_n_t<T, dim> b, int size) {
  std::fill_n(ptrC<T, dim>(b), size, a);
}

template <typename T, int dim> class SmartArray {
public:
  typedef pointer_n_t<T, dim> pointer_type;
  typedef pointer_n_t<T, dim - 1> value_type;
  pointer_type data = nullptr;
  std::array<int, dim> shape;

public:
  SmartArray() : data(nullptr) { shape.fill(0); }

  template <typename... Tail>
  explicit SmartArray(
      typename std::enable_if<sizeof...(Tail) + 1 == dim, int>::type head,
      Tail... tail)
      : shape{head, tail...} {
    if (size() > 0) {
      data = createC<T>(head, tail...);
    } else {
      data = nullptr;
    }
  }

  template <typename... Tail>
  void reallocate(
      typename std::enable_if<sizeof...(Tail) + 1 == dim, int>::type head,
      Tail... tail) {
    SmartArray<T, dim> newArray(head, tail...);
    this->swap(newArray);
  }

  decltype(shape) get_shape() const { return shape; }

  // Constructor/Assignment that allows move semantics
  SmartArray(SmartArray &&other) noexcept {
    if (!data) {
      shape.fill(0);
    }
    other.swap(*this);
  }
  SmartArray &operator=(SmartArray &&other) noexcept {
    if (!data) {
      shape.fill(0);
    }
    other.swap(*this);
    return *this;
  }

  // Constructor/Assignment that binds to nullptr
  // This makes usage with nullptr cleaner
  SmartArray(std::nullptr_t) : data(nullptr) { shape.fill(0); }
  SmartArray &operator=(std::nullptr_t) {
    reset();
    return *this;
  }

  SmartArray(SmartArray const &other) {
    if (other.shape[0] == 0) {
      shape.fill(0);
      data = nullptr;
    } else {
      shape = other.shape;
      data = createC<T, dim>(shape);
      copyC<T, dim>(other.data, data, other.size());
    }
  }
  SmartArray &operator=(SmartArray const &other) {
    if (&other == this) {
      return *this;
    }
    if (shape[0] != 0) {
      freeC(data);
    }
    if (other.shape[0] == 0) {
      shape.fill(0);
      data = nullptr;
    } else {
      shape = other.shape;
      data = createC<T, dim>(shape);
      copyC<T, dim>(other.data, data, other.size());
    }
    return *this;
  }

  void fill(T d) { fillC<T, dim>(d, data, size()); }

  // Const correct access owned object
  value_type &operator*() const { return *data; }

  // Access to smart pointer state
  pointer_type get() const { return data; }
  explicit operator bool() const { return data; }

  // Relase the control of the data, return must be captured
  pointer_type releas() noexcept {
    pointer_type result = nullptr;
    std::swap(result, data);
    return result;
  }

  void swap(SmartArray &other) noexcept {
    std::swap(data, other.data);
    std::swap(shape, other.shape);
  }

  void reset() {
    freeC(data);
    data = nullptr;
    shape.fill(0);
  }

  int size() const {
    int size = 1;
    for (int i = 0; i < shape.size(); ++i) {
      size *= shape[i];
    }
    return size;
  }

  ~SmartArray() { freeC(data); }

  T *get_data_ptr() { return ptrC<T, dim>(data); }

  value_type &operator[](std::size_t index) { return data[index]; }
};

template <typename T, int dim>
void swap(SmartArray<T, dim> &lhs, SmartArray<T, dim> &rhs) {
  lhs.swap(rhs);
}

} // namespace Memory

#endif