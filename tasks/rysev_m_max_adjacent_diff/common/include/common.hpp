#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace rysev_m_max_adjacent_diff {

using InType = std::vector<int>;
using OutType = std::pair<int, int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace rysev_m_max_adjacent_diff
