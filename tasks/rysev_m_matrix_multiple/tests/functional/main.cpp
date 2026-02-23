#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"
#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace rysev_m_matrix_multiple {

namespace {
std::vector<int> ReferenceMultiply(const std::vector<int> &A, const std::vector<int> &B, int size) {
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

auto GenerateTestData(int size, int seed = 42) {
  std::mt19937 gen(seed + size);
  std::uniform_int_distribution<> dis(1, 10);

  std::vector<int> A(size * size);
  std::vector<int> B(size * size);

  for (int i = 0; i < size * size; ++i) {
    A[i] = dis(gen);
    B[i] = dis(gen);
  }

  return std::make_tuple(A, B, size);
}
}  // namespace

class RysevMRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    auto data = GenerateTestData(size);
    input_data_ = data;

    const auto &A = std::get<0>(data);
    const auto &B = std::get<1>(data);
    expected_output_ = ReferenceMultiply(A, B, size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

const std::array<TestType, 4> kTestParam = {std::make_tuple(2, "2"), std::make_tuple(3, "3"), std::make_tuple(4, "4"),
                                            std::make_tuple(5, "5")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RysevMMatrMulMPI, InType>(kTestParam, PPC_SETTINGS_rysev_m_matrix_multiple),
                   ppc::util::AddFuncTask<RysevMMatrMulSEQ, InType>(kTestParam, PPC_SETTINGS_rysev_m_matrix_multiple));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RysevMRunFuncTestsProcesses::PrintFuncTestName<RysevMRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, RysevMRunFuncTestsProcesses, kGtestValues, kPerfTestName);

TEST_P(RysevMRunFuncTestsProcesses, MatmulFromGen) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace rysev_m_matrix_multiple
