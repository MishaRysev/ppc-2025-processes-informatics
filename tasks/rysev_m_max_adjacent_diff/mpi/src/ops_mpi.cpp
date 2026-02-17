#include "rysev_m_max_adjacent_diff/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"

namespace rysev_m_max_adjacent_diff {

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
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  size_t n = input.size();

  if (n < 2) {
    return false;
  }

  int vec_size = n;
  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_size = vec_size / world_size;
  int remainder = vec_size % world_size;

  std::vector<int> send_counts(world_size, base_size);
  std::vector<int> displs(world_size, 0);

  for (int i = 0; i < remainder; i++) {
    send_counts[i]++;
  }

  for (int i = 1; i < world_size; i++) {
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);

  MPI_Scatterv(input.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  int local_max_diff = -1;
  std::pair<int, int> local_result = std::make_pair(0, 0);

  if (local_size >= 2) {
    local_max_diff = std::abs(local_data[1] - local_data[0]);
    local_result = std::make_pair(local_data[0], local_data[1]);

    for (int i = 1; i < local_size - 1; i++) {
      int diff = std::abs(local_data[i + 1] - local_data[i]);
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_result = std::make_pair(local_data[i], local_data[i + 1]);
      }
    }
  }

  int prev_last = 0;

  if (rank < world_size - 1) {
    if (local_size > 0) {
      MPI_Send(&local_data.back(), 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    } else {
      int dummy = 0;
      MPI_Send(&dummy, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
  }

  if (rank > 0) {
    MPI_Recv(&prev_last, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (local_size > 0) {
      int diff = std::abs(local_data[0] - prev_last);
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_result = std::make_pair(prev_last, local_data[0]);
      }
    }
  }

  struct Result {
    int diff;
    int first;
    int second;
  };

  Result local_res = {local_max_diff, local_result.first, local_result.second};
  std::vector<Result> all_results;

  if (rank == 0) {
    all_results.resize(world_size);
  }

  MPI_Gather(&local_res, sizeof(Result), MPI_BYTE, all_results.data(), sizeof(Result), MPI_BYTE, 0, MPI_COMM_WORLD);

  int final_result[2] = {0, 0};

  if (rank == 0) {
    Result best = all_results[0];
    for (int i = 1; i < world_size; i++) {
      if (all_results[i].diff > best.diff) {
        best = all_results[i];
      }
    }
    final_result[0] = best.first;
    final_result[1] = best.second;
  }

  MPI_Bcast(final_result, 2, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = std::make_pair(final_result[0], final_result[1]);

  return true;
}

bool RysevMMaxAdjacentDiffMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
