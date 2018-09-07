/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace paddle {
namespace operators {

namespace xbyak {
  using memcpy_t = void(float* dst, const float* src, uint32_t steps);
  using memcpy_ptr_t = memcpy_t*;

  struct memcpy_vec_t : public Xbyak::CodeGenerator {
    memcpy_vec_t() {
      // rdi is dst
      // rsi is src
      // rdx is steps

      push(rbx);

      xor_(rax, rax);
      xor_(rbx, rbx);

      L("for_steps");
      {
        vmovups(zmm0, ptr [rsi+rax]);
        vmovups(ptr [rdi+rax], zmm0);
        add(rax, 64); // take next 16 float elements
        add(rbx, 1);
        cmp(rbx, rdx);
        jnz("for_steps");
      }

      pop(rbx);
      ret();
    }
  };
}

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");      // int tensor
    auto *output_t = context.Output<LoDTensor>("Out");  // float tensor
    auto *table_var = context.InputVar("W");

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
    int64_t ids_numel = ids_t->numel();

    if (table_var->IsType<LoDTensor>()) {
      auto *table_t = context.Input<LoDTensor>("W");
      int64_t row_number = table_t->dims()[0];
      int64_t row_width = table_t->dims()[1];

      auto *table = table_t->data<T>();
      auto *output = output_t->mutable_data<T>(context.GetPlace());
 
      for (int64_t i = 0; i < ids_numel; ++i) {
        PADDLE_ENFORCE_LT(ids[i], row_number);
        PADDLE_ENFORCE_GE(ids[i], 0, "ids %d", i);
      }

      if (padding_idx != kNoPadding) {
        for (int64_t i = 0; i < ids_numel; ++i) {
          if (ids[i] == padding_idx) {
            memset(output + i * row_width, 0, row_width * sizeof(T));
          } else {
            memcpy(output + i * row_width, table + ids[i] * row_width,
                   row_width * sizeof(T));
          }
        }
      } else {
        const uint32_t simd_width = 16;
        uint32_t steps = row_width / simd_width;
        uint32_t remain_offset = steps * simd_width;

        xbyak::memcpy_vec_t memcpy_vec;
        xbyak::memcpy_ptr_t f = reinterpret_cast<xbyak::memcpy_ptr_t>(memcpy_vec.getCode());


        for (int64_t i = 0; i < ids_numel; ++i) {
          uint64_t out_offset = i * row_width;
          uint64_t table_offset = ids[i] * row_width;

          T* out_ptr = output + out_offset;
          const T* table_ptr = table + table_offset;
          
          f(out_ptr, table_ptr, steps);
    
          for (uint32_t j = remain_offset; j < row_width; j++) {
            *(out_ptr + j) = *(table_ptr + j);
          }
        }
      }
    } else if (table_var->IsType<SelectedRows>()) {
      const auto &table_t = table_var->Get<SelectedRows>();
      int64_t row_width = table_t.value().dims()[1];
      const auto *table = table_t.value().data<T>();
      auto *output = output_t->mutable_data<T>(context.GetPlace());

      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_GE(ids[i], 0);
          auto id_index = table_t.Index(ids[i]);
          PADDLE_ENFORCE_GE(id_index, 0, "the input key should be exists.");
          memcpy(output + i * row_width, table + id_index * row_width,
                 row_width * sizeof(T));
        }
      }
    }
  }
};

template <typename T>
class LookupTableGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    if (table_var->IsType<LoDTensor>()) {
      table_dim = context.Input<LoDTensor>("W")->dims();
    } else if (table_var->IsType<SelectedRows>()) {
      auto *table_t = context.Input<SelectedRows>("W");
      table_dim = table_t->value().dims();
    } else {
      PADDLE_THROW(
          "The parameter W of a LookupTable "
          "must be either LoDTensor or SelectedRows");
    }

    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));

      auto *ids_data = ids->data<int64_t>();
      int64_t ids_num = ids->numel();

      framework::Vector<int64_t> new_rows;
      new_rows.reserve(ids_num);
      for (int64_t i = 0; i < ids_num; i++) {
        new_rows.push_back(ids_data[i]);
      }
      d_table->set_rows(new_rows);

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});
      d_table_value->mutable_data<T>(context.GetPlace());

      d_table->set_height(table_dim[0]);

      auto *d_output_data = d_output->data<T>();
      auto *d_table_data = d_table_value->data<T>();

      auto d_output_dims = d_output->dims();
      PADDLE_ENFORCE_EQ(
          d_table_value->dims(),
          framework::flatten_to_2d(d_output_dims, d_output_dims.size() - 1));
      memcpy(d_table_data, d_output_data, sizeof(T) * d_output->numel());
    } else {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<LoDTensor>(framework::GradVarName("W"));

      auto *ids_data = ids->data<int64_t>();

      int N = table_dim[0];
      int D = table_dim[1];

      auto *d_output_data = d_output->data<T>();
      auto *d_table_data = d_table->mutable_data<T>(context.GetPlace());

      memset(d_table_data, 0, d_table->numel() * sizeof(T));

      for (int64_t i = 0; i < ids->numel(); ++i) {
        PADDLE_ENFORCE_LT(ids_data[i], N);
        PADDLE_ENFORCE_GE(ids_data[i], 0);
        for (int j = 0; j < D; ++j) {
          d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
