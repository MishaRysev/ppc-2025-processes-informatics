#include <gtest/gtest.h>

#include <array>
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
}  // namespace

class RysevMRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    ppc::util::BaseRunFuncTests<InType, OutType, TestType>::SetUp();

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_ = GenerateMatrices(size);
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

namespace {

const std::array<TestType, 4> kTestParam = {std::make_tuple(2, "2"), std::make_tuple(3, "3"), std::make_tuple(4, "4"),
                                            std::make_tuple(5, "5")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RysevMMatrMulMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
                   ppc::util::AddFuncTask<RysevMMatrMulSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RysevMRunFuncTestsProcesses::PrintFuncTestName<RysevMRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, RysevMRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace rysev_m_matrix_multiple
