#include "memory.hpp"
#include <iomanip>
#include <iostream>

int main() {
  Memory::pointer_n_t<double, 3> a;

  a = Memory::createC<double>(2, 2, 2);

  Memory::freeC(a);

  // No matching constructor for initialization of 'Memory::SmartArray<3,
  // double>' Memory::SmartArray<3, double> c(2,2,2,3);

  // In template: static_assert failed due to requirement '4 < 4' "dimension
  // must be smaller than 4" Memory::SmartArray<4, double> c(2,2,2,3);

  Memory::SmartArray<double,3> c(1, 2, 2);
  std::cout << "c = " << c.data << std::endl;
  std::cout << "c shape = ";
  for (auto i : c.get_shape())
    std::cout << i << ",";
  std::cout << std::endl;

  c.reallocate(2, 2, 2);
  std::cout << "c shape = ";
  for (auto i : c.get_shape())
    std::cout << i << ",";
  std::cout << std::endl;

  int w = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
  for (int k = 0; k < 2; ++k) {
        c[i][j][k] = w++;
      }
    }
  }

  Memory::SmartArray<double, 3> d(std::move(c));
  std::cout << "Move construct d = " << d.data << std::endl;
  std::cout << "c valid? : " << bool(c) <<std::endl;
  std::cout << "d valid? : " << bool(d) <<std::endl;
  std::cout << "c shape = ";
  for (auto &&i : c.get_shape()) {
    std::cout << i << ", ";
  }
  std::cout <<std::endl;
  std::cout << "d shape = ";
  for (auto &&i : d.get_shape()) {
    std::cout << i << ", ";
  }
  std::cout <<std::endl;
  // c[0][0][0] = 1;  //segmentation fault
  for (int k = 0; k < 2; ++k) {
    std::cout << "Page " << k << std::endl;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        std::cout << std::setw(3) << d[i][j][k] << "  ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Copy construct" <<std::endl;
  Memory::SmartArray<double, 3> g = d;
  std::cout << "d shape = ";
  for (auto &&i : d.get_shape()) {
    std::cout << i << ", ";
  }
  std::cout << "g shape: ";
  for (auto &&i : g.get_shape()) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;


  Memory::SmartArray<double, 3> h(1,2,2);
  h = g;
  std::cout << "Copy assignment g = " << (void *)**g.data << std::endl;
  std::cout << "Copy assignment h = " << (void *)**h.data << std::endl;
  for (int k = 0; k < 2; ++k) {
    std::cout << "Page " << k << std::endl;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        std::cout << std::setw(3) << h[i][j][k] << "  ";
      }
      std::cout << std::endl;
    }
  }


  Memory::SmartArray<double, 3> l;
  std::cout << "Move assignment l = " << bool(l) << std::endl;
  std::cout << "Move assignment h = " << (void *)**h.data << std::endl;
  l = std::move(h);
  std::cout << "Move assignment l = " << (void *)**l.data << std::endl;
  std::cout << "Move assignment h = " << bool(h) << std::endl;


}