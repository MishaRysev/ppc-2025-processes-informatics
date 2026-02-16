#include "rysev_m_max_adjacent_diff/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "rysev_m_max_adjacent_diff/common/include/common.hpp"

namespace rysev_m_max_adjacent_diff {

RysevMMaxAdjacentDiffSEQ::RysevMMaxAdjacentDiffSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_pair(0, 0);
}

bool RysevMMaxAdjacentDiffSEQ::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool RysevMMaxAdjacentDiffSEQ::PreProcessingImpl() {
  GetOutput() = std::make_pair(0, 0);
  return true;
}

bool RysevMMaxAdjacentDiffSEQ::RunImpl() {
  const auto &input = GetInput();
  if (input.size() < 2) {
    return false;
  }

  int max_diff = -1;
  std::pair<int, int> result = std::make_pair(input[0], input[1]);

  for (size_t i = 0; i < input.size() - 1; ++i) {
    int diff = std::abs(input[i + 1] - input[i]);
    if (diff > max_diff) {
      max_diff = diff;
      result = std::make_pair(input[i], input[i + 1]);
    }
  }

  GetOutput() = result;
  return true;
}

bool RysevMMaxAdjacentDiffSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rysev_m_max_adjacent_diff
