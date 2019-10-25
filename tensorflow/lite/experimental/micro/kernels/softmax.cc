/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdint.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

#include "softmax.hpp"

namespace tflite {
namespace ops {
namespace micro {
namespace activations {



TfLiteStatus CalculateSoftmaxQuantParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* data) {

    TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
    if (input->type == kTfLiteUInt8) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    }

    static const int kScaledDiffIntegerBits = 5;

    tflite::PreprocessSoftmaxScaling(
        params->beta, input->params.scale, kScaledDiffIntegerBits,
        &data->input_multiplier, &data->input_left_shift);
    data->diff_min = -1.0 * tflite::CalculateInputRadius(
                                kScaledDiffIntegerBits, data->input_left_shift);

  return kTfLiteOk;
}



void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}


template <typename T>
TfLiteStatus SoftmaxQuantized(TfLiteContext* context, const TfLiteTensor* input,
                              TfLiteTensor* output, SoftmaxParams& data) {
  if (NumDimensions(input) >= 1 && NumDimensions(input) <= 4) {
      optimized_integer_ops::Softmax(data, GetTensorShape(input),
                           GetTensorData<T>(input), GetTensorShape(output),
                           GetTensorData<T>(output));
    return kTfLiteOk;
  } else {
    context->ReportError(
        context, "Only 1D, 2D, 3D and 4D tensors supported currently, got %dD.",
        NumDimensions(input));
    return kTfLiteError;
  }
}


TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  SoftmaxParams softmax_data;


  // TODO(ahentz): consider an implementation that works for many (all?)
  // dimensions.
  switch (input->type) {
    case kTfLiteInt8: {
        TF_LITE_ENSURE_STATUS(
                CalculateSoftmaxQuantParams(context, input, output, params, &softmax_data));

      if (NumDimensions(input) == 2) {
          SoftmaxQuantized<int8_t>(context, input, output, softmax_data);
        return kTfLiteOk;
      }
      context->ReportError(
          context, "Only 2D and 4D tensors supported currently, got %dD.",
          NumDimensions(input));
      return kTfLiteError;
    } break;
    default:
      context->ReportError(
          context, "Only float32, int8_t and uint8_t supported currently, got %d.",
          input->type);
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::SoftmaxPrepare,
                                 activations::SoftmaxEval};
  return &r;
}




}  // namespace micro
}  // namespace ops
}  // namespace tflite
