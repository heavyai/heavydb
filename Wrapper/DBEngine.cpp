/*
 * Copyright 2020 OmniSci, Inc.
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

#include "DBEngine.h"
#include <thrift/Thrift.h>
#include "Catalog/Catalog.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "QueryEngine/CompilationOptions.h"
#include "Import/Importer.h"
#include "Shared/Logger.h"
#include "Shared/mapdpath.h"
#include "Shared/sqltypes.h"
#include <boost/filesystem.hpp>
#include <boost/variant.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <array>

namespace EmbeddedDatabase {

    enum class ColumnType : uint32_t {eUNK, eINT, eDBL, eFLT, eSTR, eARR};

    /**
     * Cursor internal implementation
     */
    class CursorImpl : public Cursor {
    public:
        CursorImpl(std::shared_ptr<ResultSet> resultSet, std::shared_ptr<Data_Namespace::DataMgr> dataMgr)
        : m_resultSet(resultSet) 
        , m_DataMgr(dataMgr) {
        }

        size_t GetColCount() {
            return m_resultSet->colCount();
        }

        size_t GetRowCount() {
            return m_resultSet->rowCount();
        }

        Row GetNextRow() {
            auto row = m_resultSet->getNextRow(true, false);
            if (row.empty()) {
                std::cout << "Cursor::GetNextRow - Empty row" << std::endl;
                return Row();
            }
            return Row(row);
        }

        ColumnType GetColType(uint32_t nPos) {
            if (nPos < GetColCount()) {
                SQLTypeInfo typeInfo = m_resultSet->getColType(nPos);
                switch(typeInfo.get_type()) {
                    case kNUMERIC:
                    case kDECIMAL:
                    case kINT:
                    case kSMALLINT:
                    case kBIGINT:
                        return ColumnType::eINT;

                    case kDOUBLE:
                        return ColumnType::eDBL;

                    case kFLOAT:
                        return ColumnType::eFLT;

                    case kCHAR:
                    case kVARCHAR:
                    case kTEXT:
                        return ColumnType::eSTR;

                    default:
                        return ColumnType::eUNK;
                }
            }
            return ColumnType::eUNK;
        }

    private:
        std::shared_ptr<ResultSet> m_resultSet;
        std::weak_ptr<Data_Namespace::DataMgr> m_DataMgr;
    };

    /**
     * DBEngine internal implementation
     */
    class DBEngineImpl : public DBEngine {
    public:
        const int CALCITEPORT = 3279;
        const std::string OMNISCI_DEFAULT_DB = "omnisci";
        const std::string OMNISCI_ROOT_USER = "admin";
        const std::string OMNISCI_DATA_PATH = "//mapd_data";

        void Reset() {
          // TODO: Destroy all cursors in the m_Cursors
            if (m_pQueryRunner != nullptr) {
                m_pQueryRunner->reset();
            }
        }

        void ExecuteDDL(const std::string& sQuery) {
            if (m_pQueryRunner != nullptr) {
                m_pQueryRunner->runDDLStatement(sQuery);
            }
        }

        Cursor* ExecuteDML(const std::string& sQuery) {
            if (m_pQueryRunner != nullptr) {
                auto rs = m_pQueryRunner->runSQL(sQuery, ExecutorDeviceType::CPU);
                m_Cursors.emplace_back(new CursorImpl(rs, m_DataMgr));
                return m_Cursors.back();
            }
            return nullptr;
        }

        DBEngineImpl(const std::string& sBasePath) 
        : m_sBasePath(sBasePath)
        , m_pQueryRunner(nullptr) {
            if (!boost::filesystem::exists(m_sBasePath)) {
                std::cerr << "Catalog basepath " + m_sBasePath + " does not exist.\n";
                // Create database if it does not exist
            } else {
                MapDParameters mapdParms;
                std::string sDataPath = m_sBasePath + OMNISCI_DATA_PATH;
                m_DataMgr = std::make_shared<Data_Namespace::DataMgr>(sDataPath, mapdParms, false, 0);
                auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, m_sBasePath, 1024, 5000);
                auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
                sys_cat.init(m_sBasePath, m_DataMgr, {}, calcite, false, false, {});
                if (!sys_cat.getSqliteConnector()) {
                    std::cerr << "SqliteConnector is null " << std::endl;
                } else {
                    sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, m_Database); //TODO: Check
                    auto catalog = Catalog_Namespace::Catalog::get(m_sBasePath, m_Database, m_DataMgr, std::vector<LeafHostInfo>(), calcite, false);
                    sys_cat.getMetadataForUser(OMNISCI_ROOT_USER, m_User);
                    auto session = std::make_unique<Catalog_Namespace::SessionInfo>(catalog, m_User, ExecutorDeviceType::CPU, "");
                    m_pQueryRunner = QueryRunner::QueryRunner::init(session);
                }
            }
        }

    private:
        std::string m_sBasePath;
        std::shared_ptr<Data_Namespace::DataMgr> m_DataMgr;
        Catalog_Namespace::DBMetadata m_Database;
        Catalog_Namespace::UserMetadata m_User;
        QueryRunner::QueryRunner* m_pQueryRunner;
        std::vector<CursorImpl*> m_Cursors;
    };

    /********************************************* DBEngine external methods*/

    /**
     * Creates DBEngine instance
     *
     * @param sPath Path to the existing database
     */
    DBEngine* DBEngine::Create(std::string sPath) {
        return new DBEngineImpl(sPath);
    }

    /** DBEngine downcasting methods */
    inline DBEngineImpl * GetImpl(DBEngine* ptr) { return (DBEngineImpl *)ptr; }
    inline const DBEngineImpl * GetImpl(const DBEngine* ptr) { return (const DBEngineImpl *)ptr; }

    void DBEngine::Reset() {
        // TODO: Make sure that dbengine does not released twice
        DBEngineImpl* pEngine = GetImpl(this);
        pEngine->Reset();
    }

    void DBEngine::ExecuteDDL(std::string sQuery) {
        DBEngineImpl* pEngine = GetImpl(this);
        pEngine->ExecuteDDL(sQuery);
    }

    Cursor* DBEngine::ExecuteDML(std::string sQuery) {
        DBEngineImpl* pEngine = GetImpl(this);
        return pEngine->ExecuteDML(sQuery);
    }

    /********************************************* Row methods */

    Row::Row() {}

    Row::Row(std::vector<TargetValue>& row) : m_row(std::move(row)) {}

    int64_t Row::GetInt(size_t nCol) {
      if (nCol < m_row.size()) {
        const auto scalarValue = boost::get<ScalarTargetValue>(&m_row[nCol]);
        const auto value = boost::get<int64_t>(scalarValue);
        return *value;
      }
      return 0;
    }

    double Row::GetDouble(size_t nCol) {
      if (nCol < m_row.size()) {
        const auto scalarValue = boost::get<ScalarTargetValue>(&m_row[nCol]);
        const auto value = boost::get<double>(scalarValue);
        return *value;
      }
      return 0.;
    }

    std::string Row::GetStr(size_t nCol) {
      if (nCol < m_row.size()) {
        const auto scalarValue = boost::get<ScalarTargetValue>(&m_row[nCol]);
        auto value = boost::get<NullableString>(scalarValue);
        bool is_null = !value || boost::get<void*>(value);
        if (is_null) {
          return "Empty";
        } else {
          auto value_notnull = boost::get<std::string>(value);
          return *value_notnull;
        }
      }
      return "Out of range";
    }

    /********************************************* Сursor external methods*/

    /** Cursor downcasting methods */
    inline CursorImpl * GetImpl(Cursor* ptr) { return (CursorImpl *)ptr; }
    inline const CursorImpl * GetImpl(const Cursor* ptr) { return (const CursorImpl *)ptr; }

    size_t Cursor::GetColCount() {
        CursorImpl* pCursor = GetImpl(this);
        return pCursor->GetColCount();
    }

    size_t Cursor::GetRowCount() {
        CursorImpl* pCursor = GetImpl(this);
        return pCursor->GetRowCount();
    }

    Row Cursor::GetNextRow() {
        CursorImpl* pCursor = GetImpl(this);
        return pCursor->GetNextRow();
    }

    int Cursor::GetColType(uint32_t nPos) {
        CursorImpl* pCursor = GetImpl(this);
        int nColType = (int)pCursor->GetColType(nPos);
        return nColType;
    }
}
