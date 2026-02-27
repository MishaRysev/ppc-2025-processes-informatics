#include "rysev_m_shell_sort_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <queue>
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

bool RysevMShellSortMPI::RunImpl() {
  int data_size = 0;
  std::vector<int> input_data;

  if (rank_ == 0) {
    const auto &input_ref = GetInput();
    data_size = input_ref.size();
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

  if (rank_ == 0 && data_size > 0) {
    struct HeapNode {
      int value;
      int chunk_idx;
      size_t elem_idx;

      bool operator>(const HeapNode &other) const {
        return value > other.value;
      }
    };

    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> min_heap;

    for (int i = 0; i < num_procs_; ++i) {
      if (recv_counts[i] > 0) {
        int first_val = gathered_data[recv_displs[i]];
        min_heap.push({first_val, i, 0});
      }
    }

    std::vector<int> result;
    result.reserve(data_size);

    while (!min_heap.empty()) {
      HeapNode node = min_heap.top();
      min_heap.pop();
      result.push_back(node.value);

      size_t next_idx = node.elem_idx + 1;
      if (next_idx < static_cast<size_t>(recv_counts[node.chunk_idx])) {
        int next_val = gathered_data[recv_displs[node.chunk_idx] + next_idx];
        min_heap.push({next_val, node.chunk_idx, next_idx});
      }
    }

    GetOutput() = std::move(result);
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
