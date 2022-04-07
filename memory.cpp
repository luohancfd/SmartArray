#include "memory.hpp"

template <>
void SmartArray<double, 1>::transpose() {}

template <>
void SmartArray<float, 1>::transpose() {}

template <>
void SmartArray<int, 1>::transpose() {}

#ifdef __INTEL_MKL__
template <>
void SmartArray<double, 2>::transpose() {
  convertF_inplace(m_storage, shape);
  std::reverse(shape.begin(), shape.end());
}

template <>
void SmartArray<float, 2>::transpose() {
  convertF_inplace(m_storage, shape);
  std::reverse(shape.begin(), shape.end());
}
#endif