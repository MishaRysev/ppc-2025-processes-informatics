#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace rysev_m_matrix_multiple {

RysevMMatrMulMPI::RysevMMatrMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RysevMMatrMulMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);

  if (rank_ == 0) {
    const auto &input = GetInput();
    const auto &A = std::get<0>(input);
    const auto &B = std::get<1>(input);
    int size = std::get<2>(input);

    return !A.empty() && !B.empty() && size > 0 && A.size() == static_cast<size_t>(size * size) &&
           B.size() == static_cast<size_t>(size * size);
  }
  return true;
}

bool RysevMMatrMulMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);

  if (rank_ == 0) {
    const auto &input = GetInput();
    A_ = std::get<0>(input);
    B_ = std::get<1>(input);
    size_ = std::get<2>(input);
    C_.assign(size_ * size_, 0);
  }

  MPI_Bcast(&size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    A_.resize(size_ * size_);
    B_.resize(size_ * size_);
    C_.resize(size_ * size_);
  }

  return true;
}

bool RysevMMatrMulMPI::RunImpl() {
  MPI_Bcast(B_.data(), size_ * size_, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(num_procs_);
  std::vector<int> displs(num_procs_);

  int base_rows = size_ / num_procs_;
  int remainder = size_ % num_procs_;

  int offset = 0;
  for (int i = 0; i < num_procs_; ++i) {
    int proc_rows = base_rows + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * size_;
    displs[i] = offset;
    offset += send_counts[i];
  }

  local_rows_ = (rank_ < num_procs_) ? send_counts[rank_] / size_ : 0;

  if (send_counts[rank_] > 0) {
    local_A_.resize(send_counts[rank_]);
  } else {
    local_A_.resize(1);
  }

  MPI_Scatterv(rank_ == 0 ? A_.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_A_.data(),
               send_counts[rank_], MPI_INT, 0, MPI_COMM_WORLD);

  if (local_rows_ > 0) {
    local_C_.resize(local_rows_ * size_, 0);

    for (int i = 0; i < local_rows_; ++i) {
      for (int j = 0; j < size_; ++j) {
        int sum = 0;
        for (int k = 0; k < size_; ++k) {
          sum += local_A_[i * size_ + k] * B_[k * size_ + j];
        }
        local_C_[i * size_ + j] = sum;
      }
    }
  } else {
    local_C_.resize(1);
  }

  std::vector<int> recv_counts(num_procs_);
  std::vector<int> recv_displs(num_procs_);

  offset = 0;
  for (int i = 0; i < num_procs_; ++i) {
    int proc_rows = send_counts[i] / size_;
    recv_counts[i] = proc_rows * size_;
    recv_displs[i] = offset;
    offset += recv_counts[i];
  }

  MPI_Gatherv(local_C_.data(), local_rows_ * size_, MPI_INT, rank_ == 0 ? C_.data() : nullptr, recv_counts.data(),
              recv_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  if (num_procs_ > 1) {
    MPI_Bcast(C_.data(), size_ * size_, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool RysevMMatrMulMPI::PostProcessingImpl() {
  GetOutput() = C_;
  return true;
}

}  // namespace rysev_m_matrix_multiple
