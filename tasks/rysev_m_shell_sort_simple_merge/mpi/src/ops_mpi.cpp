#include "rysev_m_shell_sort_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
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

void RysevMShellSortMPI::MergeBlocks(const std::vector<int> &block_sizes, const std::vector<int> &blocks_data,
                                     const std::vector<int> &offsets, int total_elements) {
  merged_result_.clear();
  merged_result_.reserve(total_elements);

  std::vector<int> current_pos(num_procs_, 0);

  for (int k = 0; k < total_elements; ++k) {
    int best_proc = -1;
    int best_value = std::numeric_limits<int>::max();

    for (int proc = 0; proc < num_procs_; ++proc) {
      if (current_pos[proc] < block_sizes[proc]) {
        int value = blocks_data[offsets[proc] + current_pos[proc]];
        if (best_proc == -1 || value < best_value) {
          best_value = value;
          best_proc = proc;
        }
      }
    }

    merged_result_.push_back(best_value);
    ++current_pos[best_proc];
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
    return false;
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

  MPI_Scatterv(rank_ == 0 ? input_data.data() : nullptr, send_counts.data(), displs.data(), MPI_INT,
               local_block_.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (local_size > 0) {
    ShellSort(local_block_);
  }

  std::vector<int> gathered_data;
  if (rank_ == 0 && data_size > 0) {
    gathered_data.resize(data_size);
  }

  MPI_Gatherv(local_block_.data(), local_size, MPI_INT, rank_ == 0 ? gathered_data.data() : nullptr, send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0 && data_size > 0) {
    MergeBlocks(send_counts, gathered_data, displs, data_size);
    GetOutput() = merged_result_;
  }

  MPI_Bcast(merged_result_.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank_ != 0) {
    GetOutput() = merged_result_;
  }

  return true;
}

bool RysevMShellSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_shell_sort_simple_merge
