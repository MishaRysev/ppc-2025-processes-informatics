#include "rysev_m_shell_sort_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace rysev_m_shell_sort_simple_merge {

RysevMShellSortMPI::RysevMShellSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RysevMShellSortMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);

  if (rank_ == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool RysevMShellSortMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
  return true;
}

void RysevMShellSortMPI::ShellSort(std::vector<int> &arr) {
  int n = arr.size();
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = arr[i];
      int j;
      for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
        arr[j] = arr[j - gap];
      }
      arr[j] = temp;
    }
  }
}

std::vector<int> RysevMShellSortMPI::MergeSortedArrays(const std::vector<std::vector<int>> &sorted_chunks) {
  std::vector<int> result;
  std::vector<size_t> indices(sorted_chunks.size(), 0);

  while (true) {
    int min_val = std::numeric_limits<int>::max();
    int min_idx = -1;

    for (size_t i = 0; i < sorted_chunks.size(); ++i) {
      if (indices[i] < sorted_chunks[i].size() && sorted_chunks[i][indices[i]] < min_val) {
        min_val = sorted_chunks[i][indices[i]];
        min_idx = i;
      }
    }

    if (min_idx == -1) {
      break;
    }

    result.push_back(min_val);
    indices[min_idx]++;
  }

  return result;
}

bool RysevMShellSortMPI::RunImpl() {
  std::vector<int> input_data;
  int data_size = 0;

  if (rank_ == 0) {
    input_data = GetInput();
    data_size = input_data.size();
  }

  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (data_size == 0) {
    return false;
  }

  int base_size = data_size / num_procs_;
  int remainder = data_size % num_procs_;

  std::vector<int> send_counts(num_procs_);
  std::vector<int> displs(num_procs_);

  int offset = 0;
  for (int i = 0; i < num_procs_; ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += send_counts[i];
  }

  int local_size = send_counts[rank_];
  local_data_.resize(local_size);

  MPI_Scatterv(rank_ == 0 ? input_data.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_data_.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (local_size > 0) {
    ShellSort(local_data_);
  }

  std::vector<int> all_sizes(num_procs_);
  MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> all_displs(num_procs_);
  if (rank_ == 0) {
    offset = 0;
    for (int i = 0; i < num_procs_; ++i) {
      all_displs[i] = offset;
      offset += all_sizes[i];
    }
  }

  std::vector<int> gathered_data;
  if (rank_ == 0) {
    gathered_data.resize(data_size);
  }

  MPI_Gatherv(local_data_.data(), local_size, MPI_INT, rank_ == 0 ? gathered_data.data() : nullptr, all_sizes.data(),
              all_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<int> temp = gathered_data;
    ShellSort(temp);
    GetOutput() = temp;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool RysevMShellSortMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    return !GetOutput().empty();
  }
  return true;
}

}  // namespace rysev_m_shell_sort_simple_merge
