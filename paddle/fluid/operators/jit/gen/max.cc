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

#include "paddle/fluid/operators/jit/gen/max.h"
#include "paddle/fluid/operators/jit/registry.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void MaxJitCode::genCode() {
#if defined(__x86_64__)
  // calling convention RDI, RSI, RDX, RCX, R8, R9
  // XMM0-7 (ints are passed that way)
  //      RDI - Reference to Result
  //      RSI - PTR to Array
  //      RDX - Num classes

  // Regsters that need to be preserved: RBX,RBP, R12-R15
  Xbyak::util::Cpu current_cpu;
  if (current_cpu.has(Xbyak::util::Cpu::tAVX2)) {
    printf("AVX2 supported!\n");
  } else {
    printf("AVX2 not detected!\n");
  }

  mov(rcx, rdx);
  push(rbx);
  shr(rcx, 3);  // Divide by 8 (eight floats)
  shl(rdx, 2);  // num of Output elements * size of float (4)
  shl(rcx, 5);  // Trunc to 32 bytes

  // Compute partial maximums
  vpbroadcastd(ymm0, ptr[rsi]);
  xor_(rax, rax);  // Move offset for next 8 floating point values
  L("for_i");
  cmp(rax, rcx);
  jz("tail");
  vmovups(ymm1, ptr[rsi + rax]);  // A
  add(rax, 32);  // Move offset for next 8 floating point values
  vmaxps(ymm0, ymm0, ymm1);
  jmp("for_i");
  // Tail execution
  L("tail");
  sub(rdx, rcx);
  cmp(rdx, 16);
  jb("seq");
  vmovups(xmm2, ptr[rsi + rax]);  // A
  add(rax, 16);  // Move offset for next 4 floating point values
  sub(rdx, 16);
  vperm2f128(ymm2, ymm2, ymm2, 0);
  vmaxps(ymm0, ymm0, ymm2);  // partial maxes in ymm0
  L("seq");
  cmp(rdx, 0);
  jz("done");
  vpbroadcastd(ymm2, ptr[rsi + rax]);
  vmaxps(ymm0, ymm0, ymm2);  // partial maxes in ymm0
  sub(rdx, 4);
  add(rax, 4);
  jmp("seq");
  L("done");
  // Get within shortlisted buffer maximum
  vperm2f128(ymm1, ymm0, ymm0, 1);
  vmaxps(ymm0, ymm0, ymm1);  // partial maxes in ymm0
  vpermilps(xmm1, xmm0, 0x1B);
  vmaxps(ymm0, ymm0, ymm1);  // partial maxes in ymm0
  vpermilps(xmm1, xmm0, 1);
  vmaxps(ymm0, ymm0, ymm1);  // ymm0[0:31] contains global maximum
  vmovss(ptr[rdi], xmm0);    // Result <-Max(X[.])
  pop(rbx);

  printf("Generating Max Value code\n");
#else
  printf("32bit not supported\n");
#endif
  ret();
}

class MaxCreator : public JitCodeCreator<int> {
 public:
  bool UseMe(const int& attr) const override {
    return platform::MayIUse(platform::avx512f);
  }
  size_t CodeSize(const int& d) const override { return 256 * 1024; }
  std::unique_ptr<GenBase> CreateJitCode(const int& attr) const override {
    return make_unique<MaxJitCode>(attr, CodeSize(attr));
  }
};
}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;
REGISTER_JITKERNEL_GEN(kMax, gen::MaxCreator);
