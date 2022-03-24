/* GStreamer i.MX NN Inference demo plugin
 *
 * Copyright 2021 NXP
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tflite_inference.h"

// google-coral/edgetpu
#include "posenet/posenet_decoder_op.h"
#ifdef BUILD_WITH_EDGETPU
#include "edgetpu.h"
#endif

// tensorflow/lite
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>

// std
#include <map>
#include <fstream>


tflite_inference_t::tflite_inference_t()
{
}

tflite_inference_t::~tflite_inference_t()
{
  model_.reset();
  interpreter_.reset();
}

int tflite_inference_t::init(
  const std::string& model,
  int use_nnapi,
  int num_threads)
{
  // check model existence
  std::ifstream file(model);
  if (!file) {
    printf ("Failed to open %s", model.c_str());
    return ERROR;
  }

  model_ = tflite::FlatBufferModel::BuildFromFile(model.c_str());
  if (!model_) {
    printf ("Failed to mmap model %s", model.c_str());
    return ERROR;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());
#ifdef BUILD_WITH_EDGETPU
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
#endif

  tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
  if (!interpreter_) {
    printf ("Failed to construct TFLite interpreter");
    return ERROR;
  }
  bool allow_fp16 = false;
  interpreter_->SetAllowFp16PrecisionForFp32(allow_fp16);
#ifdef BUILD_WITH_EDGETPU
  // Bind edgeTpu context with interpreter.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, (TfLiteExternalContext*)edgetpu_context.get());
  interpreter_->SetNumThreads(1);// num_of_thread is ignored
#else
  interpreter_->SetNumThreads(num_threads);
#endif

  apply_delegate(use_nnapi);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    printf ("Failed to allocate TFLite tensors!");
    return ERROR;
  }

  if (verbose_) {
    tflite::PrintInterpreterState(interpreter_.get());
  }

  // initial inference test
  int width = 0;
  int height = 0;
  int channel = 0;
  std::vector<int> shape;
  get_input_tensor_shape(&shape);
  height = shape[1];
  width = shape[2];
  channel = shape[3];
  if ((width <= 0) || (height <= 0) || (channel != 3)) {
    printf("Not supported input shape");
    return ERROR;
  }
  size_t sz = 0;
  uint8_t* p = 0;
  int ret = get_input_tensor(&p, &sz);
  std::memset(p, 0, sz);
  if (interpreter_->Invoke() != kTfLiteOk) {
    printf("Failed to invoke TFLite interpreter");
    return ERROR;
  }

  return OK;
}

int tflite_inference_t::apply_delegate(
  int use_nnapi)
{
  // assume TFLite v2.0 or newer
  std::map<std::string, tflite::Interpreter::TfLiteDelegatePtr> delegates;
  if (use_nnapi == 1) {
    auto delegate = tflite::Interpreter::TfLiteDelegatePtr(tflite::NnApiDelegate(), [](TfLiteDelegate*) {});
    if (!delegate) {
      printf("NNAPI acceleration is unsupported on this platform.\n");
    } else {
      delegates.emplace("NNAPI", std::move(delegate));
    }
  } else if (use_nnapi == 2) {
    auto ext_delegate_option = TfLiteExternalDelegateOptionsDefault("/usr/lib/libvx_delegate.so");
    auto ext_delegate_ptr = TfLiteExternalDelegateCreate(&ext_delegate_option);
    auto delegate = tflite::Interpreter::TfLiteDelegatePtr(ext_delegate_ptr, [](TfLiteDelegate*) {});
    if (!delegate) {
      printf("vx-delegate backend is unsupported on this platform.");
    } else {
      delegates.emplace("vx-delegate", std::move(delegate));
    }
  }

  for (const auto& delegate : delegates) {
    if (interpreter_->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk) {
      printf("Failed to apply %s delegate.", delegate.first.c_str());
      return ERROR;
    } 
  }
  return OK;
}

int tflite_inference_t::inference(void)
{
  // tflite inference
  if (interpreter_->Invoke() != kTfLiteOk) {
    return ERROR;
  }
  
  return OK;
}

int tflite_inference_t::get_input_tensor_shape(
  std::vector<int> *shape)
{
  shape->clear();
  TfLiteIntArray *dims = interpreter_->tensor(interpreter_->inputs()[0])->dims;
  if (dims) {
    for (int i = 0; i < dims->size; i++) {
      shape->push_back(dims->data[i]);
    }
  }
  return OK;
}

int tflite_inference_t::get_input_tensor(
  uint8_t **ptr,
  size_t* sz)
{
  *ptr = typed_input_tensor<uint8_t>(0, sz);
  return OK;
}

int tflite_inference_t::setup_input_tensor(
  int frame_height,
  int frame_width,
  int frame_depth,
  uint8_t *paddr)
{
  // initial inference test
  int tensor_width = 0;
  int tensor_height = 0;
  int tensor_channels = 0;
  std::vector<int> shape;
  get_input_tensor_shape(&shape);
  tensor_height = shape[1];
  tensor_width = shape[2];
  tensor_channels = shape[3];
  if ((tensor_height != frame_height) || (tensor_width != frame_width) || (tensor_channels != frame_depth)) {
    printf("Input image size is not supported\n");
    return ERROR;
  }
  
  int ret = OK;
  size_t sz = 0;
  uint8_t *rgb = 0;
  ret = get_input_tensor(&rgb, &sz);
  
  if (ret == OK) {
	std::copy(paddr, paddr + sz, rgb);
	return OK;
  }
  else
  {
	return ERROR;
  }
}
