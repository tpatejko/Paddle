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
  // Move offset for next 8 floating point values
  xor_(rax, rax);
  L("for_i");
  cmp(rax, rcx);
  jz("tail");
  vmovaps(ymm1, ptr[rsi + rax]);  // A
  // Move offset for next 8 floating point values
  add(rax, 32);
  vmaxps(ymm0, ymm0, ymm1);
  jmp("for_i");

  // Tail execution
  L("tail");
  sub(rdx, rcx);
  cmp(rdx, 16);
  jb("seq");
  vmovaps(xmm2, ptr[rsi + rax]);  // A
  // Move offset for next 4 floating point values
  add(rax, 16);
  sub(rdx, 16);
  vperm2f128(ymm2, ymm2, ymm2, 0);
  // Partial maxes in ymm0
  vmaxps(ymm0, ymm0, ymm2);
  L("seq");
  cmp(rdx, 0);
  jz("done");
  vpbroadcastd(ymm2, ptr[rsi + rax]);
  // Partial maxes in ymm0
  vmaxps(ymm0, ymm0, ymm2);
  sub(rdx, 4);
  add(rax, 4);
  jmp("seq");
  L("done");
  // Get within shortlisted buffer maximum
  vperm2f128(ymm1, ymm0, ymm0, 1);
  // Partial maxes in ymm0
  vmaxps(ymm0, ymm0, ymm1);
  vpermilps(xmm1, xmm0, 0x1B);
  // Partial maxes in ymm0
  vmaxps(ymm0, ymm0, ymm1);
  vpermilps(xmm1, xmm0, 1);
  // ymm0[0:31] contains global maximum
  vmaxps(ymm0, ymm0, ymm1);
  vmovss(ptr[rdi], xmm0);  // Result <-Max(X[.])
  pop(rbx);

  ret();
}
}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;
REGISTER_JITKERNEL_GEN(kMax, gen::MaxJitCode);
