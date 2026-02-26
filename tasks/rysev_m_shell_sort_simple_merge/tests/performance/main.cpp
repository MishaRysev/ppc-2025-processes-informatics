#include <gtest/gtest.h>

#include <random>

#include "rysev_m_shell_sort_simple_merge/common/include/common.hpp"
#include "rysev_m_shell_sort_simple_merge/mpi/include/ops_mpi.hpp"
#include "rysev_m_shell_sort_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rysev_m_shell_sort_simple_merge {

class ShellSortPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSize_ = 10000;
  InType input_data_{};

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000);

    input_data_.resize(kSize_);
    for (int i = 0; i < kSize_; ++i) {
      input_data_[i] = dis(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> expected = input_data_;
    std::sort(expected.begin(), expected.end());
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShellSortPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RysevMShellSortMPI, RysevShellSortSEQ>(
    PPC_SETTINGS_rysev_m_shell_sort_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShellSortPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShellSortPerfTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace rysev_m_shell_sort_simple_merge
