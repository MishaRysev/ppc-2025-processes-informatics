#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace rysev_m_matrix_multiple {

using InType = std::tuple<std::vector<int>, std::vector<int>, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace rysev_m_matrix_multiple
