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

/**
 * Parses a call to a builtin function with special syntax.
 */

 SqlNode TryCast() :
 {
		 final SqlIdentifier name;
		 List<SqlNode> args = null;
		 SqlNode e = null;
		 final Span s;
		 SqlDataTypeSpec dt;
		 TimeUnit interval;
		 final TimeUnit unit;
		 final SqlNode node;
 }
 {
	//~ TRY_CAST ---------------------------------------
	(
			<TRY_CAST> { s = span(); }
			<LPAREN> e = Expression(ExprContext.ACCEPT_NON_QUERY) { args = startList(e); }
			<AS>
			(
					dt = DataType() { args.add(dt); }
			|
					<INTERVAL> e = IntervalQualifier() { args.add(e); }
			)
			<RPAREN> {
					// return SqlStdOperatorTable.CAST.createCall(s.end(this), args);
					return HeavyDBSqlOperatorTable.TRY_CAST.createCall(s.end(this), args);
			}
	)
}