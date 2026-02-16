#include "rysev_m_max_adjacent_diff/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"

namespace rysev_m_max_adjacent_diff {

struct DiffPair {
  int diff;
  int first;
  int second;
};

RysevMMaxAdjacentDiffMPI::RysevMMaxAdjacentDiffMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_pair(0, 0);
}

bool RysevMMaxAdjacentDiffMPI::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool RysevMMaxAdjacentDiffMPI::PreProcessingImpl() {
  GetOutput() = std::make_pair(0, 0);
  return true;
}

bool RysevMMaxAdjacentDiffMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  size_t n = input.size();

  if (n < 2) {
    return false;
  }

  int vec_size = n;
  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_size = vec_size / size;
  int remainder = vec_size % size;

  std::vector<int> send_counts(size, base_size);
  std::vector<int> displs(size, 0);

  for (int i = 0; i < remainder; i++) {
    send_counts[i]++;
  }

  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  std::vector<int> local_input(send_counts[rank]);

  MPI_Scatterv(input.data(), send_counts.data(), displs.data(), MPI_INT, local_input.data(), send_counts[rank], MPI_INT,
               0, MPI_COMM_WORLD);

  DiffPair local_best = {-1, 0, 0};

  for (size_t i = 0; i < local_input.size() - 1; ++i) {
    int diff = std::abs(local_input[i + 1] - local_input[i]);
    if (diff > local_best.diff) {
      local_best.diff = diff;
      local_best.first = local_input[i];
      local_best.second = local_input[i + 1];
    }
  }

  if (rank < size - 1) {
    int next_first = 0;
    MPI_Request request;
    MPI_Irecv(&next_first, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &request);

    int last_elem = local_input.back();
    MPI_Send(&last_elem, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    int diff = std::abs(next_first - last_elem);
    if (diff > local_best.diff) {
      local_best.diff = diff;
      local_best.first = last_elem;
      local_best.second = next_first;
    }
  }

  struct {
    int diff;
    int rank;
    int first;
    int second;
  } local = {local_best.diff, rank, local_best.first, local_best.second};

  struct {
    int diff;
    int rank;
    int first;
    int second;
  } global;

  MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

  if (global.rank == rank) {
    GetOutput() = std::make_pair(global.first, global.second);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool RysevMMaxAdjacentDiffMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
