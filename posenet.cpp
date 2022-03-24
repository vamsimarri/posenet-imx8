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

#include "posenet.h"

posenet_t::posenet_t()
{
}

posenet_t::~posenet_t()
{
}

int posenet_t::init(
  const std::string& model,
  int use_nnapi,
  int num_threads)
{
  return tflite_inference_t::init(model, use_nnapi, num_threads);
}

float* posenet_t::getNumPoses()
{
	size_t sz[4] = {0, };
	return (float *)(typed_output_tensor<float>(3, &sz[3]));
}

float* posenet_t::getPoseScores()
{
	size_t sz[4] = {0, };
	return (float *)(typed_output_tensor<float>(2, &sz[2]));
}

float* posenet_t::getKeypointScores()
{
	size_t sz[4] = {0, };
	return (float *)(typed_output_tensor<float>(1, &sz[1]));
}

float* posenet_t::getKeypointCoords()
{
	size_t sz[4] = {0, };
	return (float *)(typed_output_tensor<float>(0, &sz[0]));
}

