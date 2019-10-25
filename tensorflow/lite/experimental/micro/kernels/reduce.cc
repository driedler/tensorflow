
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"




namespace tflite {
namespace ops {
namespace micro {
namespace reduce {

// This file has reference implementation of reduce_* operators.
enum KernelType {
  kReference,
  kGenericOptimized,
};



constexpr const unsigned SCRACTH_TENSOR_SIZE = 4;
constexpr const unsigned RESOLVED_AXIS_SIZE = 2;
constexpr const unsigned TEMP_SUM_SIZE = 1280;

struct OpData {
  int32_t multiplier;
  int shift;
#if 0
  // The index of the temporary tensor where the quantized inputs are cached.
  //int scratch_tensor_index;
  struct
  {
      TfLiteTensor tensor;
      uint8_t dims_data[sizeof(TfLiteIntArray) + sizeof(int)];
      int32_t data[SCRACTH_TENSOR_SIZE];
  } scratch_tensor;

  struct
  {
      TfLiteTensor tensor;
      uint8_t dims_data[sizeof(TfLiteIntArray) + sizeof(int)];
      int32_t data[RESOLVED_AXIS_SIZE];
  } resolved_axis;

  struct
  {
      TfLiteTensor tensor;
      uint8_t dims_data[sizeof(TfLiteIntArray) + sizeof(int)];
      int8_t data[TEMP_SUM_SIZE];
  } temp_sum;
#endif
};

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    axis = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  TfLiteReducerParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* axis;
  TfLiteTensor* output;

};



#if 0
/*************************************************************************************************/
TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node, OpData* op_data)
{
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.axis->type, kTfLiteInt32);
  TfLiteIntArray* index_size;

  // Creates a temp index to iterate through input data.
  //TfLiteIntArrayFree(node->temporaries);

  TfLiteTensor* scratch_tensor = &op_data->scratch_tensor.tensor;
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  index_size = reinterpret_cast<TfLiteIntArray*>(op_data->scratch_tensor.dims_data);
  index_size->size = 1;
  index_size->data[0] = NumDimensions(op_context.input);
  TF_LITE_ENSURE(context, SCRACTH_TENSOR_SIZE <= index_size->data[0]);
  scratch_tensor->dims = index_size;
  scratch_tensor->data.i32 = op_data->scratch_tensor.data;

  // Creates a temp tensor to store resolved axis given input data.
  TfLiteTensor* resolved_axis = &op_data->resolved_axis.tensor;
  resolved_axis->type = kTfLiteInt32;
  index_size = reinterpret_cast<TfLiteIntArray*>(op_data->resolved_axis.dims_data);
  index_size->size = 1;
  index_size->data[0] = static_cast<int>(NumElements(op_context.axis));
  TF_LITE_ENSURE(context, RESOLVED_AXIS_SIZE <= index_size->data[0]);
  resolved_axis->dims = index_size;
  resolved_axis->data.i32 = op_data->resolved_axis.data;


  // Creates a temp tensor to store temp sums when calculating mean.
  TfLiteTensor* temp_sum = &op_data->temp_sum.tensor;
  temp_sum->type = kTfLiteInt8;
  index_size = reinterpret_cast<TfLiteIntArray*>(op_data->resolved_axis.dims_data);
  index_size->size = 1;
  index_size->data[0] = static_cast<int>(NumElements(op_context.output));
  TF_LITE_ENSURE(context, TEMP_SUM_SIZE <= index_size->data[0]);
  temp_sum->dims = index_size;
  temp_sum->data.int8 = op_data->temp_sum.data;


  return kTfLiteOk;
}
#endif

/*************************************************************************************************/
TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node, OpData* op_data)
{
 //  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node, op_data));

  // reduce_mean requires a buffer to store intermediate sum result.
  OpContext op_context(context, node);
  if (op_context.input->type == kTfLiteInt8)
  {
    const double real_multiplier =
        static_cast<double>(op_context.input->params.scale) /
        static_cast<double>(op_context.output->params.scale);
    int exponent;
    QuantizeMultiplier(real_multiplier, &op_data->multiplier, &exponent);
    op_data->shift = exponent;
  }

  return kTfLiteOk;
}

/*************************************************************************************************/
void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
}

/*************************************************************************************************/
template <KernelType kernel_type>
TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node)
{
  OpData data;
  TF_LITE_ENSURE_OK(context, PrepareMeanOrSum(context, node, &data));
  OpContext op_context(context, node);
  int num_axis = static_cast<int>(NumElements(op_context.axis));
//  TfLiteTensor* temp_index = &data.scratch_tensor.tensor;
//  TfLiteTensor* resolved_axis = &data.resolved_axis.tensor;
//  TfLiteTensor* temp_sum = &data.temp_sum.tensor;


  // From here, it uses the reference implementations.
  // TODO(b/139102329): Clean up the function signatures to merge the variations
  // and handle the specialized cases in the combined reference implementations
  // per each op.
  switch (op_context.input->type) {
    case kTfLiteInt8: {
      tflite::MeanParams op_params;
      op_params.axis_count = num_axis;
      ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
      const TfLiteTensor* input = op_context.input;
      reference_integer_ops::Mean(
          op_params, data.multiplier, data.shift, GetTensorShape(input),
          GetTensorData<int8_t>(input), op_context.input->params.zero_point,
          GetTensorShape(op_context.output),
          GetTensorData<int8_t>(op_context.output),
          op_context.output->params.zero_point);
    } break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}


} // namespace reduce






TfLiteRegistration* Register_MEAN()
{
    static TfLiteRegistration r = {nullptr, nullptr, nullptr,
                                   reduce::EvalMean<reduce::kReference>};
    return &r;
}


}  // namespace micro
}  // namespace ops
}  // namespace tflite
