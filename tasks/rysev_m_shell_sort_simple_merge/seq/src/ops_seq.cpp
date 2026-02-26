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
  return !GetInput().empty();
}

bool RysevShellSortSEQ::PreProcessingImpl() {
  GetOutput() = std::vector<int>();
  return true;
}

void RysevShellSortSEQ::ShellSort(std::vector<int>& arr) {
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

bool RysevShellSortSEQ::RunImpl() {
  auto input = GetInput();
  if (input.empty()) {
    return false;
  }

  std::vector<int> arr = input;
  ShellSort(arr);
  GetOutput() = arr;
  return true;
}

bool RysevShellSortSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace rysev_m_shell_sort_simple_merge