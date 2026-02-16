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
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  size_t n = input.size();

  if (n < 2) {
    return false;
  }

  std::pair<int, int> result = std::make_pair(0, 0);

  if (rank == 0) {
    int max_diff = -1;
    result = std::make_pair(input[0], input[1]);

    for (size_t i = 0; i < n - 1; ++i) {
      int diff = std::abs(input[i + 1] - input[i]);
      if (diff > max_diff) {
        max_diff = diff;
        result = std::make_pair(input[i], input[i + 1]);
      }
    }
  }

  int results[2] = {result.first, result.second};
  MPI_Bcast(results, 2, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = std::make_pair(results[0], results[1]);

  return true;
}

bool RysevMMaxAdjacentDiffMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
