// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/operators/jit/gen/jitcode.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {
class MaxJitCode : public JitCode {
 public:
  DECLARE_JIT_CODE(MaxJitCode);

  explicit MaxJitCode(int d /*unused*/, size_t code_size,
                      void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr) {
    this->genCode();
  }

  void genCode() override;
};
}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
