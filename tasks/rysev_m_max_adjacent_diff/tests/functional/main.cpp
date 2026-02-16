#include <gtest/gtest.h>

#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"
#include "rysev_m_max_adjacent_diff/mpi/include/ops_mpi.hpp"
#include "rysev_m_max_adjacent_diff/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace rysev_m_max_adjacent_diff {

class MaxAdjacentDiffFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "test_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_id = std::get<0>(params);

    switch (test_id) {
      case 1:
        input_data_ = {1, 2, 3, 4, 5};
        expected_output_ = std::make_pair(1, 2);
        break;
      case 2:
        input_data_ = {10, 10, 10, 100, 100};
        expected_output_ = std::make_pair(10, 100);
        break;
      case 3:
        input_data_ = {100, 90, 80, 70, 60};
        expected_output_ = std::make_pair(100, 90);
        break;
      case 4:
        input_data_ = {-10, -5, 0, 5, 10};
        expected_output_ = std::make_pair(-10, -5);
        break;
      case 5:
        input_data_ = {1, 100, 2, 99, 3};
        expected_output_ = std::make_pair(1, 100);
        break;
      case 6:
        input_data_ = {42, 100};
        expected_output_ = std::make_pair(42, 100);
        break;
      case 7:
        input_data_ = {5, 5, 5, 5, 5};
        expected_output_ = std::make_pair(5, 5);
        break;
      case 8:
        input_data_ = {1, 2, 3, 100, 4, 5, 6};
        expected_output_ = std::make_pair(3, 100);
        break;
      case 9:
        input_data_ = {10, 20, 30, 40, 50, 1000};
        expected_output_ = std::make_pair(50, 1000);
        break;
      case 10:
        input_data_ = {1, 100, 2, 3, 100, 4};
        expected_output_ = std::make_pair(1, 100);
        break;
      default:
        input_data_ = {0, 0};
        expected_output_ = std::make_pair(0, 0);
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_.first == output_data.first && expected_output_.second == output_data.second;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(MaxAdjacentDiffFuncTest, VectorMaxAdjacentDiff) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParams = {std::make_tuple(1, ""), std::make_tuple(2, ""), std::make_tuple(3, ""),
                                              std::make_tuple(4, ""), std::make_tuple(5, ""), std::make_tuple(6, ""),
                                              std::make_tuple(7, ""), std::make_tuple(8, ""), std::make_tuple(9, ""),
                                              std::make_tuple(10, "")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RysevMMaxAdjacentDiffMPI, InType>(kTestParams, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<RysevMMaxAdjacentDiffSEQ, InType>(kTestParams, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = MaxAdjacentDiffFuncTest::PrintFuncTestName<MaxAdjacentDiffFuncTest>;

INSTANTIATE_TEST_SUITE_P(MaxAdjacentDiffTests, MaxAdjacentDiffFuncTest, kGtestValues, kTestName);

}  // namespace

}  // namespace rysev_m_max_adjacent_diff
