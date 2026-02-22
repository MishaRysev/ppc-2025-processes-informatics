#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"

#include <chrono>
#include <random>
#include <vector>

namespace rysev_m_matrix_multiple {

RysevMMatrMulSEQ::RysevMMatrMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RysevMMatrMulSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &A = std::get<0>(input);
  const auto &B = std::get<1>(input);
  int size = std::get<2>(input);

  return !A.empty() && !B.empty() && size > 0 && A.size() == static_cast<size_t>(size * size) &&
         B.size() == static_cast<size_t>(size * size);
}

bool RysevMMatrMulSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  A_ = std::get<0>(input);
  B_ = std::get<1>(input);
  size_ = std::get<2>(input);

  C_.assign(size_ * size_, 0);
  return true;
}

bool RysevMMatrMulSEQ::RunImpl() {
  for (int i = 0; i < size_; ++i) {
    for (int j = 0; j < size_; ++j) {
      int sum = 0;
      for (int k = 0; k < size_; ++k) {
        sum += A_[i * size_ + k] * B_[k * size_ + j];
      }
      C_[i * size_ + j] = sum;
    }
  }
  return true;
}

bool RysevMMatrMulSEQ::PostProcessingImpl() {
  GetOutput() = C_;
  return !C_.empty();
}

}  // namespace rysev_m_matrix_multiple
