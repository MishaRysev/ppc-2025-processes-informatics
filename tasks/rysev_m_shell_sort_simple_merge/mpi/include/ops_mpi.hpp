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
  void MergeBlocks(const std::vector<int> &block_sizes, const std::vector<int> &blocks_data,
                   const std::vector<int> &offsets, int total_elements);

  int rank_;
  int num_procs_;
  std::vector<int> local_block_;
  std::vector<int> merged_result_;
};

}  // namespace rysev_m_shell_sort_simple_merge
