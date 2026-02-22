#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "example_processes/common/include/common.hpp"
#include "example_processes/mpi/include/ops_mpi.hpp"
#include "example_processes/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rysev_m_matrix_multiple {

InType GeneratePerformanceMatrices(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 5);

  int m = size;
  int k = size;
  int n = size;

  std::vector<int> A(m * k);
  std::vector<int> B(k * n);

  for (int i = 0; i < m * k; ++i) {
    A[i] = dis(gen);
  }

  for (int i = 0; i < k * n; ++i) {
    B[i] = dis(gen);
  }

  return std::make_tuple(A, B, m, k, n);
}

class RysevMPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSmallSize_ = 50;
  const int kMediumSize_ = 100;
  const int kLargeSize_ = 200;
  InType input_data_{};
  int current_size_ = 0;

  void SetUp() override {
    std::string test_mode = std::get<1>(GetParam());

    if (test_mode.find("small") != std::string::npos) {
      current_size_ = kSmallSize_;
    } else if (test_mode.find("medium") != std::string::npos) {
      current_size_ = kMediumSize_;
    } else {
      current_size_ = kLargeSize_;
    }

    input_data_ = GeneratePerformanceMatrices(current_size_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RysevMPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kSmallTask =
    ppc::util::MakePerfTask<InType, RysevMMatrMulMPI, RysevMMatrMulSEQ>("small", PPC_SETTINGS_example_processes);

const auto kMediumTask =
    ppc::util::MakePerfTask<InType, RysevMMatrMulMPI, RysevMMatrMulSEQ>("medium", PPC_SETTINGS_example_processes);

const auto kLargeTask =
    ppc::util::MakePerfTask<InType, RysevMMatrMulMPI, RysevMMatrMulSEQ>("large", PPC_SETTINGS_example_processes);

const auto kAllPerfTasks = std::tuple_cat(kSmallTask, kMediumTask, kLargeTask);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RysevMPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationPerfTests, RysevMPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace rysev_m_matrix_multiple
