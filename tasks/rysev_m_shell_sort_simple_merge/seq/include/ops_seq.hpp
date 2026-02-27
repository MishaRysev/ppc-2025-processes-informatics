#pragma once

#include "rysev_m_shell_sort_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_shell_sort_simple_merge {

class RysevShellSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RysevShellSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ShellSort(std::vector<int> &arr);
};

}  // namespace rysev_m_shell_sort_simple_merge
