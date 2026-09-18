/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CLIENTCONTEXT_H
#define CLIENTCONTEXT_H

#include "MetaClientContext.h"

using ClientContext = MetaClientContext<HeavyClient&, TTransport&>;

#endif
