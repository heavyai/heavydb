/*
 * Copyright 2022 OmniSci, Inc.
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

#ifndef OMNISCI_IMPORTUTILS_H
#define OMNISCI_IMPORTUTILS_H

#include <memory>
#include <vector>

#include "Fragmenter/Fragmenter.h"
#include "ImportExport/TypedImportBuffer.h"

namespace Catalog_Namespace {
class Catalog;
}

namespace import_export {

std::vector<std::unique_ptr<TypedImportBuffer>> fill_missing_columns(
    const Catalog_Namespace::Catalog* cat,
    Fragmenter_Namespace::InsertData& insert_data);

}  // namespace import_export

#endif  // OMNISCI_IMPORTUTILS_H
