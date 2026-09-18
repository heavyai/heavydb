/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.calcite.sql;

import java.util.List;

/**
 * Marker interface for SqlOperator subclasses that inject default arguments via
 * createCall() when optional operands are omitted (e.g. LTRIM fills in a trailing
 * space when called with one argument).
 *
 * Calcite's CallCopyingArgHandler only invokes createCall() when it copies a
 * call node. Implementing this interface lets SqlValidatorImpl force a copy
 * and therefore trigger default injection even when no other rewrite is needed.
 * requiresSpecialCreate() returns true only when the optional arg is actually absent,
 * avoiding unnecessary copies in the common case.
 */
public interface HeavyDBDefaultArgInjector {
    boolean requiresSpecialCreate(List<SqlNode> operands);
}
