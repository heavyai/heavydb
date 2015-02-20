#ifndef COLUMN_DESCRIPTOR_H
#define COLUMN_DESCRIPTOR_H

#include <cassert>
#include <string>
#include "../Shared/sqltypes.h"

/**
 * @type ColumnDescriptor
 * @brief specifies the content in-memory of a row in the column metadata table
 * 
 */

struct ColumnDescriptor {
    int tableId; 
    int columnId;
    std::string columnName;
    SQLTypeInfo columnType;
    EncodingType compression; // compression scheme 
    int comp_param; // compression parameter for certain encoding types
    std::string chunks;

    ColumnDescriptor() {}
    ColumnDescriptor(const int tableId, const int columnId, const std::string &columnName, const SQLTypeInfo columnType, const EncodingType compression, const int comp_param = 0): tableId(tableId), columnId(columnId), columnName(columnName),columnType(columnType),compression(compression),comp_param(comp_param) {
    } 

		inline bool is_varlen() const { return IS_STRING(columnType.type) && compression != kENCODING_DICT; }

		int getStorageSize() const {
			switch (columnType.type) {
				case kBOOLEAN:
					return sizeof(int8_t);
				case kSMALLINT:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(int16_t);
						case kENCODING_FIXED:
							return comp_param/8;
						case kENCODING_RL:
						case kENCODING_DIFF:
						case kENCODING_DICT:
						case kENCODING_SPARSE:
							assert(false);
						break;
					}
					break;
				case kINT:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(int32_t);
						case kENCODING_FIXED:
							return comp_param/8;
						case kENCODING_RL:
						case kENCODING_DIFF:
						case kENCODING_DICT:
						case kENCODING_SPARSE:
							assert(false);
						break;
					}
					break;
				case kBIGINT:
				case kNUMERIC:
				case kDECIMAL:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(int64_t);
						case kENCODING_FIXED:
							return comp_param/8;
						case kENCODING_RL:
						case kENCODING_DIFF:
						case kENCODING_DICT:
						case kENCODING_SPARSE:
							assert(false);
						break;
					}
					break;
				case kFLOAT:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(float);
						case kENCODING_FIXED:
						case kENCODING_RL:
						case kENCODING_DIFF:
						case kENCODING_DICT:
						case kENCODING_SPARSE:
							assert(false);
						break;
					}
					break;
				case kDOUBLE:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(double);
						case kENCODING_FIXED:
						case kENCODING_RL:
						case kENCODING_DIFF:
						case kENCODING_DICT:
						case kENCODING_SPARSE:
							assert(false);
						break;
					}
					break;
				case kTIME:
				case kTIMESTAMP:
					if (columnType.dimension > 0)
						assert(false); // not supported yet
				case kDATE:
					switch (compression) {
						case kENCODING_NONE:
							return sizeof(time_t);
						case kENCODING_FIXED:
						case kENCODING_RL:
						case kENCODING_DIFF:
						case kENCODING_DICT:
						case kENCODING_SPARSE:
							assert(false);
						break;
					}
					break;
				default:
					break;
			}
			return -1;
		}
};

#endif // COLUMN_DESCRIPTOR
