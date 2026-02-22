#include "example_processes/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace rysev_m_matrix_multiple {

RysevMMatrMulMPI::RysevMMatrMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  A_ = std::get<0>(in);
  B_ = std::get<1>(in);
  sizes_.M = std::get<2>(in);
  sizes_.K = std::get<3>(in);
  sizes_.N = std::get<4>(in);

  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RysevMMatrMulMPI::ValidationImpl() {
  if (A_.empty() || B_.empty()) {
    return false;
  }

  if (sizes_.M <= 0 || sizes_.K <= 0 || sizes_.N <= 0) {
    return false;
  }

  if (A_.size() != static_cast<size_t>(sizes_.M * sizes_.K)) {
    return false;
  }

  if (B_.size() != static_cast<size_t>(sizes_.K * sizes_.N)) {
    return false;
  }

  return true;
}

bool RysevMMatrMulMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    C_.assign(sizes_.M * sizes_.N, 0);
  }

  return true;
}

void RysevMMatrMulMPI::DistributeMatrixA(int rank, int size) {
  int base_rows = sizes_.M / size;
  int remainder = sizes_.M % size;

  send_counts_.resize(size);
  displs_.resize(size);

  for (int i = 0; i < size; ++i) {
    int proc_rows = base_rows + (i < remainder ? 1 : 0);
    send_counts_[i] = proc_rows * sizes_.K;
    displs_[i] = (i == 0) ? 0 : displs_[i - 1] + send_counts_[i - 1];
  }

  local_rows_ = send_counts_[rank] / sizes_.K;

  local_A_.resize(send_counts_[rank], 0);

  MPI_Scatterv(A_.data(), send_counts_.data(), displs_.data(), MPI_INT, local_A_.data(), send_counts_[rank], MPI_INT, 0,
               MPI_COMM_WORLD);
}

void RysevMMatrMulMPI::BroadcastMatrixB(int rank) {
  int b_size = sizes_.K * sizes_.N;

  if (rank != 0) {
    B_.resize(b_size);
  }

  MPI_Bcast(B_.data(), b_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void RysevMMatrMulMPI::ComputeLocalProduct() {
  local_C_.assign(local_rows_ * sizes_.N, 0);

  for (int i = 0; i < local_rows_; ++i) {
    for (int j = 0; j < sizes_.N; ++j) {
      int sum = 0;
      for (int k = 0; k < sizes_.K; ++k) {
        sum += local_A_[i * sizes_.K + k] * B_[k * sizes_.N + j];
      }
      local_C_[i * sizes_.N + j] = sum;
    }
  }
}

void RysevMMatrMulMPI::GatherResults(int rank, int size) {
  std::vector<int> recv_counts(size);
  std::vector<int> recv_displs(size);

  for (int i = 0; i < size; ++i) {
    int proc_rows = send_counts_[i] / sizes_.K;
    recv_counts[i] = proc_rows * sizes_.N;
    recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
  }

  if (rank == 0) {
    MPI_Gatherv(local_C_.data(), local_rows_ * sizes_.N, MPI_INT, C_.data(), recv_counts.data(), recv_displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_C_.data(), local_rows_ * sizes_.N, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0,
                MPI_COMM_WORLD);
  }
}

bool RysevMMatrMulMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 0) {
    return false;
  }

  DistributeMatrixA(rank, size);
  BroadcastMatrixB(rank);
  ComputeLocalProduct();
  GatherResults(rank, size);

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool RysevMMatrMulMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = C_;
    return !C_.empty();
  }

  return true;
}

}  // namespace rysev_m_matrix_multiple
