#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "example_processes/common/include/common.hpp"
#include "example_processes/mpi/include/ops_mpi.hpp"
#include "example_processes/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace rysev_m_matrix_multiple {

std::tuple<std::vector<int>, std::vector<int>, int, int, int> GenerateMatrices(int m, int k, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 10);

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

std::vector<int> SequentialMultiply(const std::vector<int> &A, const std::vector<int> &B, int m, int k, int n) {
  std::vector<int> C(m * n, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int sum = 0;
      for (int t = 0; t < k; ++t) {
        sum += A[i * k + t] * B[t * n + j];
      }
      C[i * n + j] = sum;
    }
  }

  return C;
}

class RysevMRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    input_data_ = GenerateMatrices(size, size, size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> A = std::get<0>(input_data_);
    std::vector<int> B = std::get<1>(input_data_);
    int m = std::get<2>(input_data_);
    int k = std::get<3>(input_data_);
    int n = std::get<4>(input_data_);

    std::vector<int> expected = SequentialMultiply(A, B, m, k, n);

    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

const std::array<TestType, 5> kTestParam = {std::make_tuple(2, "2"), std::make_tuple(3, "3"), std::make_tuple(4, "4"),
                                            std::make_tuple(5, "5"), std::make_tuple(6, "6")};

const auto kTestTasksList =
    ppc::util::AddFuncTask<RysevMMatrMulMPI, InType>(kTestParam, PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RysevMRunFuncTestsProcesses::PrintFuncTestName<RysevMRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, RysevMRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace rysev_m_matrix_multiple
