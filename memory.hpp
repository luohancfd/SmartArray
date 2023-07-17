/**
 * @file memory.hpp
 * @author Han Luo (han.luo@gmail.com)
 * @brief Utilities to create continuous memory for c type data
 * @version 1.0
 * @date 2020-04-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <algorithm>
#include <array>
#include <climits>
#include <type_traits>
#include <vector>

#include "mkl.h"

typedef long mem_int;

template <typename DT>
inline DT *createC(mem_int m) {
  return new DT[m];
}

template <typename DT>
inline void freeC(DT *array) {
  if (array) delete[] array;
}

template <typename DT>
inline DT *convertF(DT *array, mem_int m) {
  DT *array_f = new DT[m];
  std::copy(array, array + m, array_f);
  return array_f;
}

template <typename DT>
inline DT **createC(mem_int m, mem_int n) {
  DT **array = new DT *[m];
  *array = new DT[m * n];
  for (mem_int i = 1; i < m; ++i) {
    array[i] = array[i - 1] + n;
  }
  return array;
}

template <typename DT>
inline void freeC(DT **array) {
  if (array) {
    if (array[0]) delete[] array[0];
    delete[] array;
  }
}

template <typename DT>
inline DT **convertF(DT **array, mem_int m, mem_int n) {
  DT **array_f = createC<DT>(n, m);
  for (mem_int ii = 0; ii < m * n; ++ii) {
    mem_int i = ii / n;
    mem_int j = ii % n;
    array_f[j][i] = *(*array + ii);
  }
  return array_f;
}

template <>
inline double **convertF(double **array, mem_int m, mem_int n) {
  double **array_f = createC<double>(n, m);
#ifdef __INTEL_MKL__
  if (m <= INT_MAX && n <= INT_MAX) {
    mkl_domatcopy('R', 'T', (int)m, (int)n, 1.0, *array, (int)n, *array_f,
                  (int)m);
    return array_f;
  }
#endif
  for (mem_int ii = 0; ii < m * n; ++ii) {
    mem_int i = ii / n;
    mem_int j = ii % n;
    array_f[j][i] = *(*array + ii);
  }
  return array_f;
}

template <>
inline float **convertF(float **array, mem_int m, mem_int n) {
  float **array_f = createC<float>(n, m);
#ifdef __INTEL_MKL__
  if (m <= INT_MAX && n <= INT_MAX) {
    mkl_somatcopy('R', 'T', (int)m, (int)n, 1.0, *array, (int)n, *array_f,
                  (int)m);
    return array_f;
  }
#endif
  for (mem_int ii = 0; ii < m * n; ++ii) {
    mem_int i = ii / n;
    mem_int j = ii % n;
    array_f[j][i] = *(*array + ii);
  }
  return array_f;
}

template <typename DT>
inline DT ***createC(mem_int m, mem_int n, mem_int k) {
  DT ***array = new DT **[m];
  *array = new DT *[m * n];
  **array = new DT[m * n * k];

  for (mem_int i = 1; i < m; ++i) {
    array[i] = array[i - 1] + n;
    array[i][0] = array[i - 1][0] + n * k;
  }
  for (mem_int i = 0; i < m; ++i) {
    for (mem_int j = 1; j < n; j++) {
      array[i][j] = array[i][j - 1] + k;
    }
  }
  return array;
}

template <typename DT>
inline void freeC(DT ***array) {
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

template <typename DT>
inline DT ***convertF(DT ***array, mem_int m, mem_int n, mem_int l) {
  DT ***array_f = createC<DT>(l, n, m);
  for (mem_int ii = 0; ii < m * n * l; ++ii) {
    mem_int i = ii / mem_int(n * l);
    mem_int j = (ii - i * n * l) / l;
    mem_int k = ii - i * n * l - j * l;
    array_f[k][j][i] = *(**array + ii);
  }
  return array_f;
}

template <std::size_t dim>
std::array<mem_int, dim> get_index(int offset,
                                   const std::array<mem_int, dim> shape) {
  std::array<mem_int, dim> index;

  mem_int stride = 1;
  for (auto i : shape) {
    if (i != 0) stride *= i;
  }

  for (std::size_t axis = 0; axis < dim; ++axis) {
    if (shape[axis] != 0) {
      stride /= shape[axis];
      index[axis] = offset / stride;
      offset -= index[axis] * stride;
    } else {
      index[axis] = -1;
    }
  }

  return index;
}

template <std::size_t dim>
mem_int get_offset(const std::array<mem_int, dim> index,
                   const std::array<mem_int, dim> shape) {
  mem_int offset = 0;
  mem_int stride = 1;
  for (std::size_t axis = dim - 1; axis >= 0; --axis) {
    if (shape[axis] != 0) {
      offset += index[axis] * stride;
      stride *= shape[axis];
    }
  }
  return offset;
}

template <typename T, std::size_t dim>
struct pointer_n {
  static_assert(dim < 4, "dimension must be smaller than 4");
  static_assert(dim >= 0, "dimension must be greater than 0");
  typedef T type;
  typedef const T ctype;
};

template <typename T>
struct pointer_n<T, 1> {
  typedef T *type;
  typedef T const *ctype;
};

template <typename T>
struct pointer_n<T, 2> {
  typedef T **type;
  typedef T const **ctype;
};

template <typename T>
struct pointer_n<T, 3> {
  typedef T ***type;
  typedef T const ***ctype;
};

template <typename T, std::size_t dim>
using pointer_n_t = typename pointer_n<T, dim>::type;

template <typename T, std::size_t dim>
using const_pointer_n_t = typename pointer_n<T, dim>::ctype;

template <typename T, std::size_t dim,
          typename std::enable_if<(dim<1 || dim> 3), bool>::type = true>
pointer_n_t<T, dim> createC(std::array<mem_int, dim> shape) {
  return nullptr;
}

template <typename T, std::size_t dim,
          typename std::enable_if<(dim == 1), bool>::type = true>
pointer_n_t<T, dim> createC(std::array<mem_int, dim> shape) {
  return createC<T>(shape[0]);
}

template <typename T, std::size_t dim,
          typename std::enable_if<(dim == 2), bool>::type = true>
pointer_n_t<T, dim> createC(std::array<mem_int, dim> shape) {
  return createC<T>(shape[0], shape[1]);
}

template <typename T, std::size_t dim,
          typename std::enable_if<(dim == 3), bool>::type = true>
pointer_n_t<T, dim> createC(std::array<mem_int, dim> shape) {
  return createC<T>(shape[0], shape[1], shape[2]);
}

template <typename T, std::size_t dim,
          typename std::enable_if<(dim<1 || dim> 3), bool>::type = true>
pointer_n_t<T, dim> convertF(pointer_n_t<T, dim> array,
                             std::array<mem_int, dim> shape) {
  return nullptr;
}

template <typename T, std::size_t dim,
          typename std::enable_if<(dim == 1), bool>::type = true>
pointer_n_t<T, dim> convertF(pointer_n_t<T, dim> array,
                             std::array<mem_int, dim> shape) {
  return convertF<T>(array, shape[0]);
}

template <typename T, std::size_t dim,
          typename std::enable_if<(dim == 2), bool>::type = true>
pointer_n_t<T, dim> convertF(pointer_n_t<T, dim> array,
                             std::array<mem_int, dim> shape) {
  return convertF<T>(array, shape[0], shape[1]);
}

#ifdef __INTEL_MKL__
inline void convertF_inplace(pointer_n_t<double, 2> &array,
                             std::array<mem_int, 2> shape) {
  long m = shape[0], n = shape[1];
  double *AB = *array;
  if (m <= INT_MAX && n <= INT_MAX) {
    mkl_dimatcopy('R', 'T', m, shape[1], 1.0, AB, n, m);
    delete[] array;
    array = new double *[n];
    array[0] = AB;
    for (int i = 1; i < n; ++i) {
      array[i] = array[i - 1] + m;
    }
  } else {
    double **new_data = convertF(array, n, n);
    freeC(array);
    array = new_data;
  }
}
inline void convertF_inplace(pointer_n_t<float, 2> &array,
                             std::array<mem_int, 2> shape) {
  int m = shape[0], n = shape[1];
  float *AB = *array;
  if (m <= INT_MAX && n <= INT_MAX) {
    mkl_simatcopy('R', 'T', m, shape[1], 1.0, AB, n, m);
    delete[] array;
    array = new float *[n];
    array[0] = AB;
    for (int i = 1; i < n; ++i) {
      array[i] = array[i - 1] + m;
    }
  } else {
    float **new_data = convertF(array, n, n);
    freeC(array);
    array = new_data;
  }
}
#endif

template <typename T, std::size_t dim,
          typename std::enable_if<(dim == 3), bool>::type = true>
pointer_n_t<T, dim> convertF(pointer_n_t<T, dim> array,
                             std::array<mem_int, dim> shape) {
  return convertF<T>(array, shape[0], shape[1], shape[2]);
}

/**
 * @brief Get the pointer to the first data
 */
template <typename T, std::size_t dim>
T *ptrC(
    pointer_n_t<typename std::enable_if<(dim<1 || dim> 3), T>::type, dim> b) {
  return nullptr;
}

template <typename T, std::size_t dim>
T *ptrC(pointer_n_t<typename std::enable_if<(dim == 1), T>::type, dim> b) {
  return b;
}

template <typename T, std::size_t dim>
T *ptrC(pointer_n_t<typename std::enable_if<(dim == 2), T>::type, dim> b) {
  return *b;
}

template <typename T, std::size_t dim>
T *ptrC(pointer_n_t<typename std::enable_if<(dim == 3), T>::type, dim> b) {
  return **b;
}

template <typename T, std::size_t dim>
void copyC(pointer_n_t<T, dim> a, pointer_n_t<T, dim> b, mem_int size) {
  T *start = ptrC<T, dim>(a);
  T *end = ptrC<T, dim>(a) + size;
  std::copy(start, end, ptrC<T, dim>(b));
}

template <typename T, std::size_t dim>
void fillC(T a, pointer_n_t<T, dim> b, mem_int size) {
  if (size > 0) std::fill_n(ptrC<T, dim>(b), size, a);
}

template <typename T, std::size_t dim>
class SmartArray {
public:
  typedef pointer_n_t<T, dim> pointer_type;
  typedef const_pointer_n_t<T, dim> const_pointer_type;
  typedef pointer_n_t<T, dim - 1> value_type;
  typedef typename std::add_const<value_type>::type const_value_type;
  typedef T *iterator;
  typedef T data_type;

  /**
   * @brief The underlying array serving as element storage.
   *        The higher dimension varies faster
   *        i.e. if the array shape is m_storage[3][2]
   *        The memory layout will be
   *        (0,0), (0,1), (1,0), (1,1)
   *
   */
  pointer_type m_storage =
      nullptr;  // m_storage[][][] is used to access the array like a matrix
  std::array<mem_int, dim> shape;

public:
  SmartArray() : m_storage(nullptr) { shape.fill(0); }

  SmartArray(const std::array<mem_int, dim> &_shape) : shape(_shape) {
    m_storage = createC<T, dim>(shape);
  }

  SmartArray(const std::array<mem_int, dim> &_shape, pointer_type m_storage_)
      : shape(_shape), m_storage(m_storage_){};

  template <typename... Tail>
  explicit SmartArray(
      typename std::enable_if<sizeof...(Tail) + 1 == dim, mem_int>::type head,
      Tail... tail)
      : shape{head, tail...} {
    if (size() > 0) {
      m_storage = createC<T>(head, tail...);
    } else {
      m_storage = nullptr;
    }
  }

  template <typename... Tail>
  void reallocate(
      typename std::enable_if<sizeof...(Tail) + 1 == dim, mem_int>::type head,
      Tail... tail) {
    if (size() > 0) freeC(m_storage);
    shape = {head, tail...};
    if (size() > 0)
      m_storage = createC<T>(head, tail...);
    else
      m_storage = nullptr;
  }

  template <typename T2>
  void reallocate(T2 *new_shape) {
    if (size() > 0) freeC(m_storage);
    for (std::size_t i = 0; i < dim; ++i) {
      shape[i] = new_shape[i];
    }
    if (size() > 0)
      m_storage = createC<T, dim>(shape);
    else
      m_storage = nullptr;
  }

  void reallocate(decltype(shape) &new_shape) { reallocate(new_shape.data()); }

  decltype(shape) get_shape() const { return shape; }

  // Constructor/Assignment that allows move semantics
  SmartArray(SmartArray &&other) noexcept {
    if (!m_storage) {
      shape.fill(0);
    }
    other.swap(*this);
  }

  SmartArray &operator=(SmartArray &&other) noexcept {
    if (!m_storage) {
      shape.fill(0);
    }
    other.swap(*this);
    return *this;
  }

  // Constructor/Assignment that binds to nullptr
  // This makes usage with nullptr cleaner
  SmartArray(std::nullptr_t) : m_storage(nullptr) { shape.fill(0); }
  SmartArray &operator=(std::nullptr_t) {
    reset();
    return *this;
  }

  SmartArray(SmartArray const &other) {
    if (other.shape[0] == 0) {
      shape.fill(0);
      m_storage = nullptr;
    } else {
      shape = other.shape;
      m_storage = createC<T, dim>(shape);
      copyC<T, dim>(other.m_storage, m_storage, other.size());
    }
  }

  SmartArray &operator=(SmartArray const &other) {
    if (&other == this) {
      return *this;
    }
    if (other.shape[0] == 0) {
      shape = other.shape;
      freeC(m_storage);
      m_storage = nullptr;
    } else {
      if (shape != other.shape) {
        freeC(m_storage);
        shape = other.shape;
        m_storage = createC<T, dim>(shape);
      }
      copyC<T, dim>(other.m_storage, m_storage, other.size());
    }
    return *this;
  }

  template <typename T2>
  SmartArray &operator=(const T2 &value) {
    fill((T)value);
    return *this;
  }

  SmartArray<T, dim> transpose_copy() const {
    SmartArray<T, dim> temp;
    std::reverse_copy(shape.begin(), shape.end(), temp.shape.begin());
    temp.m_storage = convertF<T>(this->m_storage, this->shape);
    return temp;
  }

  /**
   * @brief Transpose the underlying data.
   * @note The transpose doesn't occur inplace. Thus if you get a pointer
   *       to m_storage before, these is no guarantee the pointer will be
   *       the same.
   * @note this->data() will only be the same if T == double/float
   *       and dim == 1/2
   */
  void transpose() {
    decltype(m_storage) new_m_storage = convertF<T>(m_storage, shape);
    freeC(m_storage);
    m_storage = new_m_storage;
    std::reverse(shape.begin(), shape.end());
  }

  void fill(const T &d) { fillC<T, dim>(d, m_storage, size()); }

  /**
   * @brief Return a pointer to the managed multidimensional array
   *
   * @return const_pointer_type
   */
  const_pointer_type cget() const {
    return const_cast<const_pointer_type>(m_storage);
  }

  /**
   * @brief Return a pointer to the managed multidimensional array
   *
   * @return pointer_type
   */
  pointer_type get() { return m_storage; }

  explicit operator bool() const { return m_storage; }

  // Release the control of the data, return must be captured
  pointer_type release() noexcept {
    pointer_type result = nullptr;
    std::swap(result, m_storage);
    return result;
  }

  void swap(SmartArray &other) noexcept {
    std::swap(m_storage, other.m_storage);
    std::swap(shape, other.shape);
  }

  void reset() {
    freeC(m_storage);
    m_storage = nullptr;
    shape.fill(0);
  }

  mem_int size() const {
    mem_int size = 1;
    for (auto i : shape) {
      size *= i;
    }
    return size;
  }

  ~SmartArray() { freeC(m_storage); }

  /**
   * @brief Returns pointer to the underlying array serving as element storage
   *        The pointer is such that range [data(); data() + size()) is always
   *        a valid range, even if the container is empty (data() is not
   *        dereferenceable in that case).
   *
   * @return T*
   */
  T *data() {
    if (m_storage) return ptrC<T, dim>(m_storage);
    return nullptr;
  }

  T const *data() const { return data(); }

  T *begin() { return data(); }

  T *end() { return begin() + size(); }

  T const *cbegin() const { return ptrC<T, dim>(m_storage); }

  T const *cend() const { return ptrC<T, dim>(m_storage) + size(); }

  T *at_ptr(const std::array<mem_int, dim> &element_index) {
    mem_int offset = 0;
    mem_int stride = 1;
    for (std::size_t i = dim; i-- > 0;) {
      offset += element_index[i] * stride;
      stride *= shape[i];
    }
    return begin() + offset;
  }

  T const *at_ptr(const std::array<mem_int, dim> &element_index) const {
    mem_int offset = 0;
    mem_int stride = 1;
    for (std::size_t i = dim; i-- > 0;) {
      offset += element_index[i] * stride;
      stride *= shape[i];
    }

    return const_cast<T const *>(cbegin() + offset);
  }

  T &at(const std::array<mem_int, dim> &element_index) {
    return *at_ptr(element_index);
  }

  const T &at(const std::array<mem_int, dim> &element_index) const {
    return *at_ptr(element_index);
  }

  template <typename... varg>
  T &at(typename std::enable_if<sizeof...(varg) + 1 == dim, mem_int>::type head,
        varg... tail) {
    std::array<mem_int, dim> element_index{head, tail...};
    return at(element_index);
  }

  template <typename... varg>
  const T &at(
      typename std::enable_if<sizeof...(varg) + 1 == dim, mem_int>::type head,
      varg... tail) const {
    std::array<mem_int, dim> element_index{head, tail...};
    return at(element_index);
  }

  T min() const {
    auto min_element = std::min_element(cbegin(), cend());
    return *min_element;
  }

  T max() const {
    auto max_element = std::max_element(cbegin(), cend());
    return *max_element;
  }

  /**
   * @brief Return the minimum along a given axis, similar to numpy.ndarray.min
   *
   * @param axis
   * @return SmartArray<T, dim - 1>
   */
  SmartArray<T, dim - 1> min(const mem_int axis) const {
    auto n_elem = size();
    std::vector<T *> rp(n_elem, nullptr);
    std::array<mem_int, dim - 1> new_shape;
    {
      std::size_t i = 0;
      for (std::size_t j = 0; j < dim; ++j) {
        if (axis != j) {
          new_shape[i] = shape[j];
          ++i;
        }
      }
    }

    mem_int stride = 1;
    for (std::size_t i = axis + 1; i < dim; ++i) {
      stride *= shape[i];
    }

    SmartArray<T, dim - 1> r(new_shape);
    auto it2 = r.begin();
    auto it = begin();

    auto shift_shape = shape;
    shift_shape[axis] = 0;

    for (mem_int offset = 0; offset < n_elem; ++offset) {
      auto index = get_index<dim>(offset, shape);
      auto offset2 = get_offset<dim>(index, shift_shape);
      // auto index2 = get_index<dim-1>(offset2, new_shape);
      if (index[axis] == 0 || *(it2 + offset2) > *(it + offset)) {
        *(it2 + offset2) = *(it + offset);
      }
    }
    return r;
  }

  /**
   * @brief Return the maximum along a given axis, similar to numpy.ndarray.max
   *
   * @param axis
   * @return SmartArray<T, dim - 1>
   */
  SmartArray<T, dim - 1> max(const mem_int axis) const {
    auto n_elem = size();
    std::vector<T *> rp(n_elem, nullptr);
    std::array<mem_int, dim - 1> new_shape;
    {
      std::size_t i = 0;
      for (std::size_t j = 0; j < dim; ++j) {
        if (axis != j) {
          new_shape[i] = shape[j];
          ++i;
        }
      }
    }

    mem_int stride = 1;
    for (std::size_t i = axis + 1; i < dim; ++i) {
      stride *= shape[i];
    }

    SmartArray<T, dim - 1> r(new_shape);
    auto it2 = r.begin();
    auto it = begin();

    auto shift_shape = shape;
    shift_shape[axis] = 0;

    for (mem_int offset = 0; offset < n_elem; ++offset) {
      auto index = get_index<dim>(offset, shape);
      auto offset2 = get_offset<dim>(index, shift_shape);
      // auto index2 = get_index<dim-1>(offset2, new_shape);
      if (index[axis] == 0 || *(it2 + offset2) < *(it + offset)) {
        *(it2 + offset2) = *(it + offset);
      }
    }
    return r;
  }

  value_type &operator[](std::size_t index) { return m_storage[index]; }

  const_value_type &operator[](std::size_t index) const {
    return m_storage[index];
  }

#define SmartArrayOperatorInc(MATH_OP)                                      \
  SmartArray<T, dim> &operator MATH_OP##=(const SmartArray<T, dim> other) { \
    T *a = this->data();                                                    \
    T const *b = other.data();                                              \
    auto n = this->size();                                                  \
    for (decltype(n) i = 0; i < n; ++i) {                                   \
      a[i] MATH_OP## = b[i];                                                \
    }                                                                       \
    return *this;                                                           \
  }                                                                         \
  SmartArray<T, dim> &operator MATH_OP##=(const T value) {                  \
    T *a = this->data();                                                    \
    auto n = this->size();                                                  \
    for (decltype(n) i = 0; i < n; ++i) {                                   \
      a[i] MATH_OP## = value;                                               \
    }                                                                       \
    return *this;                                                           \
  }

  // clang-format off
  SmartArrayOperatorInc(+)
  SmartArrayOperatorInc(-)
  SmartArrayOperatorInc(*)
  SmartArrayOperatorInc(/)
  // clang-format on
};

template <>
void SmartArray<double, 1>::transpose();

template <>
void SmartArray<float, 1>::transpose();

template <>
void SmartArray<mem_int, 1>::transpose();

#ifdef __INTEL_MKL__
template <>
void SmartArray<double, 2>::transpose();

template <>
void SmartArray<float, 2>::transpose();
#endif

template <typename T, std::size_t dim>
void swap(SmartArray<T, dim> &lhs, SmartArray<T, dim> &rhs) {
  lhs.swap(rhs);
}

// Han: after benchmark I didn't see clear advantage of MKL operator
#define SmartArrayOperatorMKL(MKL_DOUBLE_OP, MKL_FLOAT_OP, MATH_OP) \
  template <std::size_t dim>                                        \
  SmartArray<double, dim> operator MATH_OP(                         \
      SmartArray<double, dim> &__restrict__ lhs,                    \
      SmartArray<double, dim> &__restrict__ rhs) {                  \
    SmartArray<double, dim> result(lhs.shape);                      \
    mem_int n = lhs.size();                                         \
    if (n < INT_MAX) {                                              \
      MKL_DOUBLE_OP((int)n, lhs.data(), rhs.data(), result.data()); \
    } else {                                                        \
      double *a = lhs.data();                                       \
      double *b = rhs.data();                                       \
      double *c = result.data();                                    \
      for (decltype(n) i = 0; i < n; ++i) {                         \
        c[i] = a[i] MATH_OP b[i];                                   \
      }                                                             \
    }                                                               \
    return result;                                                  \
  }                                                                 \
  template <std::size_t dim>                                        \
  SmartArray<float, dim> operator MATH_OP(                          \
      SmartArray<float, dim> &__restrict__ lhs,                     \
      SmartArray<float, dim> &__restrict__ rhs) {                   \
    SmartArray<float, dim> result(lhs.shape);                       \
    mem_int n = lhs.size();                                         \
    if (n < INT_MAX) {                                              \
      MKL_FLOAT_OP((int)n, lhs.data(), rhs.data(), result.data());  \
    } else {                                                        \
      double *a = lhs.data();                                       \
      double *b = rhs.data();                                       \
      double *c = result.data();                                    \
      for (decltype(n) i = 0; i < n; ++i) {                         \
        c[i] = a[i] MATH_OP b[i];                                   \
      }                                                             \
    }                                                               \
    return result;                                                  \
  }

#define SmartArrayOperator(MATH_OP)                                           \
  template <typename T, std::size_t dim>                                      \
  SmartArray<T, dim> operator MATH_OP(SmartArray<T, dim> &__restrict__ lhs,   \
                                      SmartArray<T, dim> &__restrict__ rhs) { \
    SmartArray<T, dim> result(lhs.shape);                                     \
    auto n = lhs.size();                                                      \
    T *a = lhs.data();                                                        \
    T *b = rhs.data();                                                        \
    T *c = result.data();                                                     \
    for (decltype(n) i = 0; i < n; ++i) {                                     \
      c[i] = a[i] MATH_OP b[i];                                               \
    }                                                                         \
    return result;                                                            \
  }

// clang-format off
SmartArrayOperator(-)
SmartArrayOperator(+)
SmartArrayOperator(*)
SmartArrayOperator(/)
// clang-format on

#endif
