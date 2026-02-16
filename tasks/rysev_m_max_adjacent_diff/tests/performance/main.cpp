#include <gtest/gtest.h>

#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"
#include "rysev_m_max_adjacent_diff/mpi/include/ops_mpi.hpp"
#include "rysev_m_max_adjacent_diff/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rysev_m_max_adjacent_diff {

class MaxAdjacentDiffPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kVectorSize_ = 1000000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kVectorSize_);
    for (int i = 0; i < kVectorSize_; ++i) {
      input_data_[i] = i;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.first >= 0 && output_data.second >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MaxAdjacentDiffPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RysevMMaxAdjacentDiffMPI, RysevMMaxAdjacentDiffSEQ>(
    PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaxAdjacentDiffPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaxAdjacentDiffPerfTest, kGtestValues, kPerfTestName);

}  // namespace rysev_m_max_adjacent_diff
