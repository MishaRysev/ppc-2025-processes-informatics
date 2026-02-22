#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"
#include "rysev_m_matrix_multiple/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rysev_m_matrix_multiple {

class RysevMRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data > 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

const std::array<TestType, 3> kPerfSizes = {std::make_tuple(50, "50"), std::make_tuple(100, "100"),
                                            std::make_tuple(150, "150")};

const auto kMPITasks = ppc::util::AddFuncTask<RysevMMatrMulMPI, InType>(kPerfSizes, PPC_SETTINGS_example_processes);

const auto kSEQTasks = ppc::util::AddFuncTask<RysevMMatrMulSEQ, InType>(kPerfSizes, PPC_SETTINGS_example_processes);

const auto kAllPerfTasks = std::tuple_cat(kMPITasks, kSEQTasks);

const auto kGtestValues = ppc::util::ExpandToValues(kAllPerfTasks);

const auto kPerfTestName = RysevMRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationPerfTests, RysevMRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace rysev_m_matrix_multiple
