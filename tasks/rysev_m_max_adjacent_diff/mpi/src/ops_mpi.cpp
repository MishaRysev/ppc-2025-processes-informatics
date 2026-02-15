#include "rysev_m_max_adjacent_diff/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"
#include "util/include/util.hpp"

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
  return (GetInput().size() >= 2);
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

  size_t total_pairs = n - 1;
  size_t pairs_per_process = total_pairs / size;
  size_t remainder = total_pairs % size;

  size_t start_pair = rank * pairs_per_process + std::min(static_cast<size_t>(rank), remainder);
  size_t end_pair = start_pair + pairs_per_process + (static_cast<size_t>(rank) < remainder ? 1 : 0);

  DiffPair local_best = {-1, 0, 0};

  for (size_t i = start_pair; i < end_pair && i < total_pairs; ++i) {
    int diff = std::abs(input[i + 1] - input[i]);
    if (diff > local_best.diff) {
      local_best.diff = diff;
      local_best.first = input[i];
      local_best.second = input[i + 1];
    }
  }

  DiffPair global_best = {-1, 0, 0};

  MPI_Op max_diff_op;
  MPI_Op_create([](void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
    DiffPair *in = static_cast<DiffPair *>(invec);
    DiffPair *inout = static_cast<DiffPair *>(inoutvec);
    for (int i = 0; i < *len; i++) {
      if (in[i].diff > inout[i].diff) {
        inout[i] = in[i];
      }
    }
  }, 1, &max_diff_op);

  MPI_Allreduce(&local_best, &global_best, 1, MPI_BYTE, max_diff_op, MPI_COMM_WORLD);
  MPI_Op_free(&max_diff_op);

  GetOutput() = std::make_pair(global_best.first, global_best.second);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool RysevMMaxAdjacentDiffMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
