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

  int is_valid = 1;
  if (rank_ == 0) {
    is_valid = GetInput().empty() ? 0 : 1;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool RysevMShellSortMPI::PreProcessingImpl() {
  return true;
}

void RysevMShellSortMPI::ShellSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
}

bool RysevMShellSortMPI::RunImpl() {
  int data_size = 0;
  std::vector<int> input_data;

  if (rank_ == 0) {
    const auto &input_ref = GetInput();
    data_size = static_cast<int>(input_ref.size());
    if (data_size > 0) {
      input_data.assign(input_ref.begin(), input_ref.end());
    }
  }

  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (data_size == 0) {
    GetOutput().clear();
    return true;
  }

  std::vector<int> send_counts(num_procs_, 0);
  std::vector<int> displs(num_procs_, 0);

  if (rank_ == 0) {
    int base = data_size / num_procs_;
    int remainder = data_size % num_procs_;
    int offset = 0;
    for (int i = 0; i < num_procs_; ++i) {
      send_counts[i] = base + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += send_counts[i];
    }
  }

  MPI_Bcast(send_counts.data(), num_procs_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), num_procs_, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = send_counts[rank_];
  local_block_.clear();
  if (local_size > 0) {
    local_block_.resize(local_size);
  }

  int dummy = 0;
  int *local_ptr = local_size > 0 ? local_block_.data() : &dummy;

  MPI_Scatterv(rank_ == 0 ? input_data.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_ptr,
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (local_size > 0) {
    ShellSort(local_block_);
  }

  std::vector<int> gathered_data;
  if (rank_ == 0) {
    gathered_data.resize(data_size);
  }

  MPI_Gatherv(local_ptr, local_size, MPI_INT, rank_ == 0 ? gathered_data.data() : nullptr, send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<int> result;
    result.reserve(data_size);
    std::vector<int> indices(num_procs_, 0);

    for (int i = 0; i < data_size; ++i) {
      int best_proc = -1;
      int best_val = 0;
      for (int j = 0; j < num_procs_; ++j) {
        if (indices[j] < send_counts[j]) {
          best_proc = j;
          best_val = gathered_data[displs[j] + indices[j]];
          break;
        }
      }
      for (int j = best_proc + 1; j < num_procs_; ++j) {
        if (indices[j] < send_counts[j]) {
          int val = gathered_data[displs[j] + indices[j]];
          if (val < best_val) {
            best_val = val;
            best_proc = j;
          }
        }
      }
      result.push_back(best_val);
      ++indices[best_proc];
    }

    GetOutput() = std::move(result);
  }

  if (rank_ != 0) {
    GetOutput().resize(data_size);
  }
  MPI_Bcast(GetOutput().data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool RysevMShellSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_shell_sort_simple_merge
