#include "rysev_m_shell_sort_simple_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace rysev_m_shell_sort_simple_merge {

RysevShellSortSEQ::RysevShellSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool RysevShellSortSEQ::ValidationImpl() {
  return true;
}

bool RysevShellSortSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>();
  return true;
}

void RysevShellSortSEQ::ShellSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
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

bool RysevShellSortSEQ::RunImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  std::vector<int> arr;
  arr.reserve(input.size());
  arr.assign(input.begin(), input.end());

  ShellSort(arr);
  GetOutput() = std::move(arr);
  return true;
}

bool RysevShellSortSEQ::PostProcessingImpl() {
  return !GetOutput().empty() || GetInput().empty();
}

}  // namespace rysev_m_shell_sort_simple_merge
