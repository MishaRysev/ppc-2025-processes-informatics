#pragma once

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_max_adjacent_diff {

class RysevMMaxAdjacentDiffSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RysevMMaxAdjacentDiffSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rysev_m_max_adjacent_diff
