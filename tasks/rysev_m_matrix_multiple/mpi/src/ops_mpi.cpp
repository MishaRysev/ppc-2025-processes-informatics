bool RysevMMatrMulMPI::RunImpl() {
  MPI_Bcast(B_.data(), size_ * size_, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(num_procs_);
  std::vector<int> displs(num_procs_);

  int base_rows = size_ / num_procs_;
  int remainder = size_ % num_procs_;

  int offset = 0;
  for (int i = 0; i < num_procs_; ++i) {
    int proc_rows = base_rows + (i < remainder ? 1 : 0);
    send_counts[i] = proc_rows * size_;
    displs[i] = offset;
    offset += send_counts[i];
  }

  local_rows_ = send_counts[rank_] / size_;

  size_t local_a_size = send_counts[rank_] > 0 ? send_counts[rank_] : 1;
  local_A_.resize(local_a_size);
  if (send_counts[rank_] == 0) {
  }

  size_t local_c_size = (local_rows_ * size_) > 0 ? local_rows_ * size_ : 1;
  local_C_.assign(local_c_size, 0);

  MPI_Scatterv(rank_ == 0 ? A_.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_A_.data(),
               send_counts[rank_], MPI_INT, 0, MPI_COMM_WORLD);

  if (local_rows_ > 0) {
    for (int i = 0; i < local_rows_; ++i) {
      for (int j = 0; j < size_; ++j) {
        int sum = 0;
        for (int k = 0; k < size_; ++k) {
          sum += local_A_[i * size_ + k] * B_[k * size_ + j];
        }
        local_C_[i * size_ + j] = sum;
      }
    }
  }

  std::vector<int> recv_counts(num_procs_);
  std::vector<int> recv_displs(num_procs_);

  offset = 0;
  for (int i = 0; i < num_procs_; ++i) {
    int proc_rows = send_counts[i] / size_;
    recv_counts[i] = proc_rows * size_;
    recv_displs[i] = offset;
    offset += recv_counts[i];
  }

  MPI_Gatherv(local_C_.data(), local_rows_ * size_, MPI_INT, rank_ == 0 ? C_.data() : nullptr, recv_counts.data(),
              recv_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}
