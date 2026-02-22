#pragma once

#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_matrix_multiple {

class RysevMMatrMulSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit RysevMMatrMulSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> A_;
  std::vector<int> B_;
  std::vector<int> C_;
  int size_;
};

}  // namespace rysev_m_matrix_multiple
