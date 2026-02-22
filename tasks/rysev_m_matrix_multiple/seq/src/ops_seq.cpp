#include "example_processes/seq/include/ops_seq.hpp"

#include <vector>

namespace rysev_m_matrix_multiple {

RysevMMatrMulSEQ::RysevMMatrMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  A_ = std::get<0>(in);
  B_ = std::get<1>(in);
  sizes_.M = std::get<2>(in);
  sizes_.K = std::get<3>(in);
  sizes_.N = std::get<4>(in);

  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RysevMMatrMulSEQ::ValidationImpl() {
  if (sizes_.M <= 0 || sizes_.K <= 0 || sizes_.N <= 0) {
    return false;
  }

  if (A_.size() != static_cast<size_t>(sizes_.M * sizes_.K)) {
    return false;
  }

  if (B_.size() != static_cast<size_t>(sizes_.K * sizes_.N)) {
    return false;
  }

  return true;
}

bool RysevMMatrMulSEQ::PreProcessingImpl() {
  C_.assign(sizes_.M * sizes_.N, 0);
  return true;
}

bool RysevMMatrMulSEQ::RunImpl() {
  if (sizes_.M == 0 || sizes_.K == 0 || sizes_.N == 0) {
    return false;
  }

  for (int i = 0; i < sizes_.M; ++i) {
    for (int j = 0; j < sizes_.N; ++j) {
      int sum = 0;
      for (int k = 0; k < sizes_.K; ++k) {
        sum += A_[i * sizes_.K + k] * B_[k * sizes_.N + j];
      }
      C_[i * sizes_.N + j] = sum;
    }
  }

  return true;
}

bool RysevMMatrMulSEQ::PostProcessingImpl() {
  GetOutput() = C_;
  return !C_.empty();
}

}  // namespace rysev_m_matrix_multiple
