#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace rysev_m_matrix_multiple {

using MatrixData = std::tuple<std::vector<int>, std::vector<int>, int, int, int>;

using InType = MatrixData;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

struct MatrixSizes {
  int M;
  int K;
  int N;

  MatrixSizes() : M(0), K(0), N(0) {}
  MatrixSizes(int m, int k, int n) : M(m), K(k), N(n) {}
};

}  // namespace rysev_m_matrix_multiple
