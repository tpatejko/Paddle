// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/fc_gru_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void BuildFCGRUPattern(PDPattern* pattern) {
  // Create Operators
  std::cout << "Dupa" << std::endl;

  pattern->NewNode("mul")->assert_is_op("mul");
  pattern->NewNode("elementwise_add")->assert_is_op("elementwise_add");
  pattern->NewNode("gru")->assert_is_op("gru");
}

std::unique_ptr<ir::Graph> FCGruFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  std::cout << "GRU transpiler" << std::endl;
  //  FusePassBase::Init("fc_gru_pass", graph.get());

  GraphPatternDetector gpd;
  BuildFCGRUPattern(gpd.mutable_pattern());

  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_gru_pass, paddle::framework::ir::FCGruFusePass);
