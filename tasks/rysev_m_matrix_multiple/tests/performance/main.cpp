#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <string>
#include <tuple>
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

class RysevMRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_ = GeneratePerformanceMatrices(size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

const std::array<TestType, 3> kPerfSizes = {std::make_tuple(50, "50"), std::make_tuple(100, "100"),
                                            std::make_tuple(150, "150")};

const auto kMPITasks = ppc::util::AddFuncTask<RysevMMatrMulMPI, InType>(kPerfSizes, PPC_SETTINGS_example_processes);

const auto kSEQTasks = ppc::util::AddFuncTask<RysevMMatrMulSEQ, InType>(kPerfSizes, PPC_SETTINGS_example_processes);

const auto kAllPerfTasks = std::tuple_cat(kMPITasks, kSEQTasks);

const auto kGtestValues = ppc::util::ExpandToValues(kAllPerfTasks);

const auto kPerfTestName = RysevMRunPerfTestsProcesses::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationPerfTests, RysevMRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace rysev_m_matrix_multiple
