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
  int rank;
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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  size_t n = input.size();

  if (n < 2) {
    return false;
  }

  int vec_size = n;
  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(size, 0);
  std::vector<int> displs(size, 0);

  if (rank == 0) {
    int base_size = vec_size / size;
    int remainder = vec_size % size;

    for (int i = 0; i < size; i++) {
      send_counts[i] = base_size + (i < remainder ? 1 : 0);
    }

    displs[0] = 0;
    for (int i = 1; i < size; i++) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  MPI_Bcast(send_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(send_counts[rank]);
  MPI_Scatterv(input.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[rank], MPI_INT,
               0, MPI_COMM_WORLD);

  DiffPair local_best = {-1, 0, 0, rank};

  for (size_t i = 0; i < local_data.size() - 1; i++) {
    int diff = std::abs(local_data[i + 1] - local_data[i]);
    if (diff > local_best.diff) {
      local_best.diff = diff;
      local_best.first = local_data[i];
      local_best.second = local_data[i + 1];
      local_best.rank = rank;
    }
  }

  if (rank > 0) {
    int prev_last;
    MPI_Recv(&prev_last, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int diff = std::abs(local_data[0] - prev_last);
    if (diff > local_best.diff) {
      local_best.diff = diff;
      local_best.first = prev_last;
      local_best.second = local_data[0];
      local_best.rank = rank;
    }
  }

  if (rank < size - 1) {
    MPI_Send(&local_data.back(), 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  }

  struct {
    int diff;
    int rank;
    int first;
    int second;
  } global_best;

  MPI_Allreduce(&local_best, &global_best, 1, MPI_INT64_T, MPI_MAXLOC, MPI_COMM_WORLD);

  if (global_best.rank == rank) {
    GetOutput() = std::make_pair(global_best.first, global_best.second);
  }

  return true;
}

bool RysevMMaxAdjacentDiffMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
