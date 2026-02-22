#include "rysev_m_matrix_multiple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <chrono>
#include <numeric>
#include <random>
#include <vector>

namespace rysev_m_matrix_multiple {

RysevMMatrMulMPI::RysevMMatrMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RysevMMatrMulMPI::ValidationImpl() {
  return GetInput() > 0;
}

bool RysevMMatrMulMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    int size = GetInput();

    data_.M = size;
    data_.K = size;
    data_.N = size;

    data_.A.resize(data_.M * data_.K);
    data_.B.resize(data_.K * data_.N);

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < data_.M * data_.K; ++i) {
      data_.A[i] = dis(gen);
    }

    for (int i = 0; i < data_.K * data_.N; ++i) {
      data_.B[i] = dis(gen);
    }

    C_.assign(data_.M * data_.N, 0);
  }

  return true;
}

bool RysevMMatrMulMPI::RunImpl() {
  int sizes[3] = {data_.M, data_.K, data_.N};
  MPI_Bcast(sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    data_.M = sizes[0];
    data_.K = sizes[1];
    data_.N = sizes[2];
  }

  std::vector<int> send_counts(size_);
  std::vector<int> displs(size_);

  int base_rows = data_.M / size_;
  int remainder = data_.M % size_;

  int offset = 0;
  for (int i = 0; i < size_; ++i) {
    int proc_rows = base_rows + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * data_.K;
    displs[i] = offset;
    offset += send_counts[i];
  }

  local_rows_ = send_counts[rank_] / data_.K;
  local_A_.resize(send_counts[rank_]);

  MPI_Scatterv(rank_ == 0 ? data_.A.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_A_.data(),
               send_counts[rank_], MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    data_.B.resize(data_.K * data_.N);
  }
  MPI_Bcast(data_.B.data(), data_.K * data_.N, MPI_INT, 0, MPI_COMM_WORLD);

  local_C_.assign(local_rows_ * data_.N, 0);

  for (int i = 0; i < local_rows_; ++i) {
    for (int j = 0; j < data_.N; ++j) {
      int sum = 0;
      for (int k = 0; k < data_.K; ++k) {
        sum += local_A_[i * data_.K + k] * data_.B[k * data_.N + j];
      }
      local_C_[i * data_.N + j] = sum;
    }
  }

  std::vector<int> recv_counts(size_);
  std::vector<int> recv_displs(size_);

  offset = 0;
  for (int i = 0; i < size_; ++i) {
    int proc_rows = send_counts[i] / data_.K;
    recv_counts[i] = proc_rows * data_.N;
    recv_displs[i] = offset;
    offset += recv_counts[i];
  }

  if (rank_ == 0) {
    MPI_Gatherv(local_C_.data(), local_rows_ * data_.N, MPI_INT, C_.data(), recv_counts.data(), recv_displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_C_.data(), local_rows_ * data_.N, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool RysevMMatrMulMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    int sum = 0;
    for (int val : C_) {
      sum += val;
    }
    GetOutput() = sum;
  }
  return true;
}

}  // namespace rysev_m_matrix_multiple
