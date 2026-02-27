#pragma once

#include <vector>

#include "rysev_m_shell_sort_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_shell_sort_simple_merge {

class RysevMShellSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RysevMShellSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ShellSort(std::vector<int> &arr);

  int rank_;
  int num_procs_;
  std::vector<int> local_block_;
};

}  // namespace rysev_m_shell_sort_simple_merge
