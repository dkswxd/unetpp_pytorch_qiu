#include <torch/extension.h>

#include "pytorch_cpp_helper.hpp"

void DeformConvForwardCUDAKernelLauncher(Tensor input, Tensor weight,
                                         Tensor offset, Tensor output,
                                         Tensor columns, Tensor ones, int kW,
                                         int kH, int dW, int dH, int padW,
                                         int padH, int dilationW, int dilationH,
                                         int group, int deformable_group,
                                         int im2col_step);

void DeformConvBackwardInputCUDAKernelLauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradInput,
    Tensor gradOffset, Tensor weight, Tensor columns, int kW, int kH, int dW,
    int dH, int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, int im2col_step);

void DeformConvBackwardParametersCUDAKernelLauncher(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    float scale, int im2col_step);

void deform_conv_forward_cuda(Tensor input, Tensor weight, Tensor offset,
                              Tensor output, Tensor columns, Tensor ones,
                              int kW, int kH, int dW, int dH, int padW,
                              int padH, int dilationW, int dilationH, int group,
                              int deformable_group, int im2col_step) {
  DeformConvForwardCUDAKernelLauncher(
      input, weight, offset, output, columns, ones, kW, kH, dW, dH, padW, padH,
      dilationW, dilationH, group, deformable_group, im2col_step);
}

void deform_conv_backward_input_cuda(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradInput,
                                     Tensor gradOffset, Tensor weight,
                                     Tensor columns, int kW, int kH, int dW,
                                     int dH, int padW, int padH, int dilationW,
                                     int dilationH, int group,
                                     int deformable_group, int im2col_step) {
  DeformConvBackwardInputCUDAKernelLauncher(
      input, offset, gradOutput, gradInput, gradOffset, weight, columns, kW, kH,
      dW, dH, padW, padH, dilationW, dilationH, group, deformable_group,
      im2col_step);
}

void deform_conv_backward_parameters_cuda(
    Tensor input, Tensor offset, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    float scale, int im2col_step) {
  DeformConvBackwardParametersCUDAKernelLauncher(
      input, offset, gradOutput, gradWeight, columns, ones, kW, kH, dW, dH,
      padW, padH, dilationW, dilationH, group, deformable_group, scale,
      im2col_step);
}


void deform_conv_forward(Tensor input, Tensor weight, Tensor offset,
                         Tensor output, Tensor columns, Tensor ones, int kW,
                         int kH, int dW, int dH, int padW, int padH,
                         int dilationW, int dilationH, int group,
                         int deformable_group, int im2col_step) {
  if (input.device().is_cuda()) {

    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);

    deform_conv_forward_cuda(input, weight, offset, output, columns, ones, kW,
                             kH, dW, dH, padW, padH, dilationW, dilationH,
                             group, deformable_group, im2col_step);
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}

void deform_conv_backward_input(Tensor input, Tensor offset, Tensor gradOutput,
                                Tensor gradInput, Tensor gradOffset,
                                Tensor weight, Tensor columns, int kW, int kH,
                                int dW, int dH, int padW, int padH,
                                int dilationW, int dilationH, int group,
                                int deformable_group, int im2col_step) {
  if (input.device().is_cuda()) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradInput);
    CHECK_CUDA_INPUT(gradOffset);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(columns);

    deform_conv_backward_input_cuda(input, offset, gradOutput, gradInput,
                                    gradOffset, weight, columns, kW, kH, dW, dH,
                                    padW, padH, dilationW, dilationH, group,
                                    deformable_group, im2col_step);
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}

void deform_conv_backward_parameters(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradWeight,
                                     Tensor columns, Tensor ones, int kW,
                                     int kH, int dW, int dH, int padW, int padH,
                                     int dilationW, int dilationH, int group,
                                     int deformable_group, float scale,
                                     int im2col_step) {
  if (input.device().is_cuda()) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradWeight);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);

    deform_conv_backward_parameters_cuda(input, offset, gradOutput, gradWeight,
                                         columns, ones, kW, kH, dW, dH, padW,
                                         padH, dilationW, dilationH, group,
                                         deformable_group, scale, im2col_step);
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}



void ModulatedDeformConvForwardCUDAKernelLauncher(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias);

void ModulatedDeformConvBackwardCUDAKernelLauncher(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

void modulated_deform_conv_forward_cuda(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  ModulatedDeformConvForwardCUDAKernelLauncher(
      input, weight, bias, ones, offset, mask, output, columns, kernel_h,
      kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}

void modulated_deform_conv_backward_cuda(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  ModulatedDeformConvBackwardCUDAKernelLauncher(
      input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
      grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
      deformable_group, with_bias);
}

void modulated_deform_conv_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  if (input.device().is_cuda()) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(bias);
    CHECK_CUDA_INPUT(ones);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(mask);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(columns);

    modulated_deform_conv_forward_cuda(
        input, weight, bias, ones, offset, mask, output, columns, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias);
  } else {
    AT_ERROR("ModulatedDeformConv is not implemented on CPU");
  }
}

void modulated_deform_conv_backward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  if (input.device().is_cuda()) {
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(bias);
    CHECK_CUDA_INPUT(ones);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(mask);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(grad_input);
    CHECK_CUDA_INPUT(grad_weight);
    CHECK_CUDA_INPUT(grad_bias);
    CHECK_CUDA_INPUT(grad_offset);
    CHECK_CUDA_INPUT(grad_mask);
    CHECK_CUDA_INPUT(grad_output);

    modulated_deform_conv_backward_cuda(
        input, weight, bias, ones, offset, mask, columns, grad_input,
        grad_weight, grad_bias, grad_offset, grad_mask, grad_output, kernel_h,
        kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, with_bias);
  } else {
    AT_ERROR("ModulatedDeformConv is not implemented on CPU");
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward (CUDA)");
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input (CUDA)");
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters (CUDA)");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward (CUDA)");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward (CUDA)");
}
