#include <gtest/gtest.h>

#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"
#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rysev_m_matrix_multiple {

namespace {
InType GeneratePerfMatrices(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 5);

  std::vector<int> A(size * size);
  std::vector<int> B(size * size);

  for (int i = 0; i < size * size; ++i) {
    A[i] = dis(gen);
    B[i] = dis(gen);
  }

  return std::make_tuple(A, B, size);
}
}  // namespace

class RysevMRunPerfTestsProcesses : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    int size = GetParam();
    input_data_ = GeneratePerfMatrices(size);

    mpi_task_ = std::make_shared<RysevMMatrMulMPI>(input_data_);
    seq_task_ = std::make_shared<RysevMMatrMulSEQ>(input_data_);
  }

  InType input_data_;
  std::shared_ptr<RysevMMatrMulMPI> mpi_task_;
  std::shared_ptr<RysevMMatrMulSEQ> seq_task_;
};

TEST_P(RysevMRunPerfTestsProcesses, MPI) {
  ASSERT_TRUE(mpi_task_->Validation());

  ASSERT_TRUE(mpi_task_->PreProcessing());

  ASSERT_TRUE(mpi_task_->Run());

  ASSERT_TRUE(mpi_task_->PostProcessing());

  auto output = mpi_task_->GetOutput();
  ASSERT_FALSE(output.empty());
}

TEST_P(RysevMRunPerfTestsProcesses, SEQ) {
  ASSERT_TRUE(seq_task_->Validation());

  ASSERT_TRUE(seq_task_->PreProcessing());

  ASSERT_TRUE(seq_task_->Run());

  ASSERT_TRUE(seq_task_->PostProcessing());

  auto output = seq_task_->GetOutput();
  ASSERT_FALSE(output.empty());
}

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationPerfTests, RysevMRunPerfTestsProcesses, ::testing::Values(50, 100, 150),
                         [](const testing::TestParamInfo<int> &info) { return "Size_" + std::to_string(info.param); });

}  // namespace rysev_m_matrix_multiple
