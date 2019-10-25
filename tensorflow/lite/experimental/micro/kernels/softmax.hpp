#pragma once

#include <algorithm>
#include <stdint.h>
#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"


namespace tflite {
namespace optimized_integer_ops {


// Quantized softmax with int8 input and output.
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const int8* input_data,
                    const RuntimeShape& output_shape, int8* output_data) {
  const int32 input_beta_multiplier = params.input_multiplier;
  const int32 input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int b = 0; b < outer_size; ++b) {
    const int8* input_data_ptr = input_data + b * depth;
    int8* output_data_ptr = output_data + b * depth;

    // Determine the largest entry in the current row
    int8_t max_in_row = -128;
    {
      int c = 0;
      for (; c < depth; ++c) {
        max_in_row = std::max(max_in_row, input_data_ptr[c]);
      }
    }


    // Compute the sum of exponentials of the differences of entries in the
    // current row from the largest entry in the current row.
    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    {
      int c = 0;
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          sum_of_exps =
              sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                exp_on_negative_values(scaled_diff_f8));
        }
      }
    }

    // Compute the fixed-point multiplier and shift that we need to apply to
    // perform a division by the above-computed sum-of-exponentials.
    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    // Compute the quotients of exponentials of differences of entries in the
    // current row from the largest entry, over the previously-computed sum of
    // exponentials.
    {
      int c = 0;
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);

          FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
          const int32 unsat_output = gemmlowp::RoundingDivideByPOT(
              (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);
          const int32 shifted_output = unsat_output - 128;

          output_data_ptr[c] =
              static_cast<int8>(std::max(std::min(shifted_output, 127), -128));

        } else {
          output_data_ptr[c] = -128;
        }
      }
    }
  }
}


}}
