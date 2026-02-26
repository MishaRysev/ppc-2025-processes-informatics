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
  if (sorted_chunks.empty()) {
    return std::vector<int>();
  }

  std::vector<int> result;
  size_t total_size = 0;
  for (const auto &chunk : sorted_chunks) {
    total_size += chunk.size();
  }
  result.reserve(total_size);

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
  int data_size = 0;
  std::vector<int> input_data;

  if (rank_ == 0) {
    const auto &input_ref = GetInput();
    data_size = input_ref.size();

    if (data_size > 0) {
      input_data.reserve(data_size);
      for (int i = 0; i < data_size; ++i) {
        input_data.push_back(input_ref[i]);
      }
    }
  }

  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (data_size == 0) {
    return false;
  }

  std::vector<int> send_counts(num_procs_, 0);
  std::vector<int> displs(num_procs_, 0);

  int base_size = data_size / num_procs_;
  int remainder = data_size % num_procs_;

  int offset = 0;
  for (int i = 0; i < num_procs_; ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += send_counts[i];
  }

  int local_size = send_counts[rank_];

  local_data_.clear();
  if (local_size > 0) {
    local_data_.resize(local_size);
  }

  MPI_Scatterv(rank_ == 0 ? input_data.data() : nullptr, send_counts.data(), displs.data(), MPI_INT,
               local_size > 0 ? local_data_.data() : nullptr, local_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (local_size > 0) {
    ShellSort(local_data_);
  }

  std::vector<int> recv_counts(num_procs_, 0);
  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> recv_displs(num_procs_, 0);
  if (rank_ == 0) {
    offset = 0;
    for (int i = 0; i < num_procs_; ++i) {
      recv_displs[i] = offset;
      offset += recv_counts[i];
    }
  }

  std::vector<int> gathered_data;
  if (rank_ == 0 && data_size > 0) {
    gathered_data.resize(data_size);
  }

  MPI_Gatherv(local_size > 0 ? local_data_.data() : nullptr, local_size, MPI_INT,
              rank_ == 0 ? gathered_data.data() : nullptr, recv_counts.data(), recv_displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  // Финальное слияние на процессе 0
  if (rank_ == 0) {
    if (data_size > 0 && !gathered_data.empty()) {
      std::vector<std::vector<int>> chunks;
      chunks.reserve(num_procs_);

      for (int i = 0; i < num_procs_; ++i) {
        if (recv_counts[i] > 0) {
          std::vector<int> chunk;
          chunk.reserve(recv_counts[i]);
          for (int j = 0; j < recv_counts[i]; ++j) {
            chunk.push_back(gathered_data[recv_displs[i] + j]);
          }
          chunks.push_back(std::move(chunk));
        }
      }

      if (!chunks.empty()) {
        std::vector<int> result = MergeSortedArrays(chunks);
        GetOutput().clear();
        GetOutput().reserve(result.size());
        for (size_t i = 0; i < result.size(); ++i) {
          GetOutput().push_back(result[i]);
        }
      }
    }
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
