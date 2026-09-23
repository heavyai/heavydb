#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SHADER_ARTIFACTS=~/shader_artifacts

cp $SHADER_ARTIFACTS/MSMultiGpuComposite.frag.builder ./Renderer/Inputs/MSMultiGpuComposite.frag.builder
cp $SHADER_ARTIFACTS/fullScreenTriangle.vert.builder ./Renderer/Inputs/fullScreenTriangle.vert.builder
cp $SHADER_ARTIFACTS/SMAABlendingWeightCalculation.frag.builder ./Renderer/Inputs/SMAABlendingWeightCalculation.frag.builder
cp $SHADER_ARTIFACTS/SMAAEdgeDetection.frag.builder ./Renderer/Inputs/SMAAEdgeDetection.frag.builder
cp $SHADER_ARTIFACTS/SMAANeighborhoodBlending.frag.builder ./Renderer/Inputs/SMAANeighborhoodBlending.frag.builder
cp $SHADER_ARTIFACTS/SMAAPassThru.vert.builder ./Renderer/Inputs/SMAAPassThru.vert.builder
cp $SHADER_ARTIFACTS/SeparateMultiSample.frag.builder ./Renderer/Inputs/SeparateMultiSample.frag.builder

cp $SHADER_ARTIFACTS/fastSymbolTemplate.frag.builder ./Manual/Inputs/fastSymbolTemplate.frag.builder
cp $SHADER_ARTIFACTS/fastSymbolTemplate.vert.builder ./Manual/Inputs/fastSymbolTemplate.vert.builder
cp $SHADER_ARTIFACTS/lineTemplate.frag.builder ./Manual/Inputs/lineTemplate.frag.builder
cp $SHADER_ARTIFACTS/lineTemplate.geom.builder ./Manual/Inputs/lineTemplate.geom.builder
cp $SHADER_ARTIFACTS/lineTemplate.vert.builder ./Manual/Inputs/lineTemplate.vert.builder
cp $SHADER_ARTIFACTS/pointTemplate.frag.builder ./Manual/Inputs/pointTemplate.frag.builder
cp $SHADER_ARTIFACTS/pointTemplate.vert.builder ./Manual/Inputs/pointTemplate.vert.builder
