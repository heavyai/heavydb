<#--
 Copyright 2022 HEAVY.AI, Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
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