#include <gtest/gtest.h>

#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"
#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

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

class RysevMRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  using TestParamType =
      typename ::testing::TestWithParam<typename ppc::util::BaseRunPerfTests<InType, OutType>::ParamType>::ParamType;

  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() {
    ppc::util::BaseRunPerfTests<InType, OutType>::SetUp();
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(this->GetParam());
    int size = std::get<0>(params);
    input_data_ = GeneratePerfMatrices(size);
  }

  bool CheckTestOutputData(OutType &output_data) {
    return !output_data.empty();
  }

  InType GetTestInputData() {
    return input_data_;
  }

 private:
  InType input_data_;
};

const std::array<TestType, 3> kPerfSizes = {std::make_tuple(50, "50"), std::make_tuple(100, "100"),
                                            std::make_tuple(150, "150")};

const auto kMPITasks =
    ppc::util::AddFuncTask<RysevMMatrMulMPI, InType, TestType>(kPerfSizes, PPC_SETTINGS_example_processes);
const auto kSEQTasks =
    ppc::util::AddFuncTask<RysevMMatrMulSEQ, InType, TestType>(kPerfSizes, PPC_SETTINGS_example_processes);
const auto kAllPerfTasks = std::tuple_cat(kMPITasks, kSEQTasks);
const auto kGtestValues = ppc::util::ExpandToValues(kAllPerfTasks);
const auto kPerfTestName = RysevMRunPerfTestsProcesses::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationPerfTests, RysevMRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace rysev_m_matrix_multiple
