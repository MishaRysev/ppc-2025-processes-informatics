#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "rysev_m_shell_sort_simple_merge/common/include/common.hpp"
#include "rysev_m_shell_sort_simple_merge/mpi/include/ops_mpi.hpp"
#include "rysev_m_shell_sort_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace rysev_m_shell_sort_simple_merge {

class RysevMShellSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    std::mt19937 gen(42 + size);
    std::uniform_int_distribution<> dis(1, 1000);

    input_data_.resize(size);
    for (int i = 0; i < size; ++i) {
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

 private:
  InType input_data_;
};

namespace {

TEST_P(RysevMShellSortFuncTests, ShellSortFromRandom) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(10, "10"), std::make_tuple(50, "50"),
                                            std::make_tuple(100, "100"), std::make_tuple(500, "500"),
                                            std::make_tuple(1000, "1000")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RysevMShellSortMPI, InType>(kTestParam, PPC_SETTINGS_rysev_m_shell_sort_simple_merge),
    ppc::util::AddFuncTask<RysevShellSortSEQ, InType>(kTestParam, PPC_SETTINGS_rysev_m_shell_sort_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RysevMShellSortFuncTests::PrintFuncTestName<RysevMShellSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShellSortTests, RysevMShellSortFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace rysev_m_shell_sort_simple_merge
