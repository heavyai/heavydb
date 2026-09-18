<#--
 SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

/*
 * Kill either running or pending query using the following syntax:
 *
 * KILL QUERY <querySession>
 */
SqlDdl SqlKillQuery(Span s) :
{
    SqlNode querySession;
}
{
    <KILL> <QUERY>
    querySession = StringLiteral()
    {
        return new SqlKillQuery(s.end(this), querySession.toString());
    }
}