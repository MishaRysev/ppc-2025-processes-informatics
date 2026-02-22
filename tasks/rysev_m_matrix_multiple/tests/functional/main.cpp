#include <gtest/gtest.h>

#include <array>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"
#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"

namespace rysev_m_matrix_multiple {

namespace {
InType GenerateMatrices(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 10);

  std::vector<int> A(size * size);
  std::vector<int> B(size * size);

  for (int i = 0; i < size * size; ++i) {
    A[i] = dis(gen);
    B[i] = dis(gen);
  }

  return std::make_tuple(A, B, size);
}

std::vector<int> MultiplyMatrices(const std::vector<int> &A, const std::vector<int> &B, int size) {
  std::vector<int> C(size * size, 0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int sum = 0;
      for (int k = 0; k < size; ++k) {
        sum += A[i * size + k] * B[k * size + j];
      }
      C[i * size + j] = sum;
    }
  }
  return C;
}
}  // namespace

class RysevMMatrixMulTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    size_ = GetParam();
    input_data_ = GenerateMatrices(size_);

    mpi_task_ = std::make_shared<RysevMMatrMulMPI>(input_data_);
    seq_task_ = std::make_shared<RysevMMatrMulSEQ>(input_data_);

    const auto &A = std::get<0>(input_data_);
    const auto &B = std::get<1>(input_data_);
    expected_ = MultiplyMatrices(A, B, size_);
  }

  int size_;
  InType input_data_;
  std::vector<int> expected_;
  std::shared_ptr<RysevMMatrMulMPI> mpi_task_;
  std::shared_ptr<RysevMMatrMulSEQ> seq_task_;
};

TEST_P(RysevMMatrixMulTest, TestMPI) {
  ASSERT_TRUE(mpi_task_->Validation());
  ASSERT_TRUE(mpi_task_->PreProcessing());
  ASSERT_TRUE(mpi_task_->Run());
  ASSERT_TRUE(mpi_task_->PostProcessing());

  auto output = mpi_task_->GetOutput();
  ASSERT_EQ(output, expected_);
}

TEST_P(RysevMMatrixMulTest, TestSEQ) {
  ASSERT_TRUE(seq_task_->Validation());
  ASSERT_TRUE(seq_task_->PreProcessing());
  ASSERT_TRUE(seq_task_->Run());
  ASSERT_TRUE(seq_task_->PostProcessing());

  auto output = seq_task_->GetOutput();
  ASSERT_EQ(output, expected_);
}

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, RysevMMatrixMulTest, ::testing::Values(2, 3, 4, 5),
                         [](const testing::TestParamInfo<int> &info) { return "Size_" + std::to_string(info.param); });

}  // namespace rysev_m_matrix_multiple
