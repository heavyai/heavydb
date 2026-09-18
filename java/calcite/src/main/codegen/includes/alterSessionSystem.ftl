<#--
 SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

SqlDdl SqlAlterSystem(Span s) :
{
    SqlDdl alterSystem;
}
{
    <ALTER>
    <SYSTEM>
    (
      alterSystem = SqlAlterSystemClear(s)
    |
      alterSystem = SqlAlterSystemControlExecutorQueue(s)
    )
    {
        return alterSystem;
    }
}

/*
 * Clear CPU or GPU memory
 *
 * ALTER SYSTEM CLEAR CPU|GPU|RENDER MEMORY
 */
SqlDdl SqlAlterSystemClear(Span s) :
{
    String cacheType;
}
{
    <CLEAR>
    (
        <CPU>
        {
            cacheType = "CPU";
        }
    |
        <GPU>
        {
            cacheType = "GPU";
        }
    |
        <RENDER>
        {
            cacheType = "RENDER";
        }
    )
    <MEMORY>
    {
        return new SqlAlterSystemClear(s.end(this), cacheType);
    }
}

/*
 * Pause/Resume Executor Queue
 *
 * ALTER SYSTEM PAUSE|RESUME EXECUTOR QUEUE
 */
SqlDdl SqlAlterSystemControlExecutorQueue(Span s) :
{
    String queueAction;
}
{
    (
      <PAUSE>
      {
          queueAction = "PAUSE";
      }
    |
      <RESUME>
      {
          queueAction = "RESUME";
      }
    )
    <EXECUTOR>
    <QUEUE>
    {
        return new SqlAlterSystemControlExecutorQueue(s.end(this), queueAction);
    }
}

/*
 * Set a session's parameter
 *
 * ALTER SESSION SET parameter='string' or
 * ALTER SESSION SET parameter='numeric';
 */
SqlDdl SqlAlterSessionSet(Span s) :
{
    final SqlIdentifier sessionParameter;
    final SqlNode parameterValue;
}
{
    <ALTER>
    <SESSION>
    <SET>
    sessionParameter = CompoundIdentifier()
    <EQ>
    (
        parameterValue = StringLiteral()
    |
        parameterValue = NumericLiteral()
    )
    {
        return new SqlAlterSessionSet(s.end(this), sessionParameter.toString(),parameterValue.toString().replaceAll("^\'|\'$", ""));
    }
}