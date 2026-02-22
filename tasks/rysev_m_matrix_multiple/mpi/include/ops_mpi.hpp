#pragma once

#include <vector>

#include "example_processes/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_matrix_multiple {

class RysevMMatrMulMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RysevMMatrMulMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeMatrixA(int rank, int size);
  void BroadcastMatrixB(int rank);
  void ComputeLocalProduct();
  void GatherResults(int rank, int size);

  std::vector<int> A_;
  std::vector<int> B_;
  MatrixSizes sizes_;

  std::vector<int> local_A_;
  std::vector<int> local_C_;
  int local_rows_;

  std::vector<int> C_;

  std::vector<int> send_counts_;
  std::vector<int> displs_;
};

}  // namespace rysev_m_matrix_multiple
