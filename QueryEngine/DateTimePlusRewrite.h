/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef QUERYENGINE_DATETIMEPLUSREWRITE_H
#define QUERYENGINE_DATETIMEPLUSREWRITE_H

#include <memory>

namespace Analyzer {

class Expr;

class FunctionOper;

}  // namespace Analyzer

std::shared_ptr<Analyzer::Expr> rewrite_to_date_trunc(const Analyzer::FunctionOper*);

#endif  // QUERYENGINE_DATETIMEPLUSREWRITE_H
