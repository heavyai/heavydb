#ifndef FRONTEND_VIEW_DESCRIPTOR_H
#define FRONTEND_VIEW_DESCRIPTOR_H

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"

/**
 * @type FrontendViewDescriptor
 * @brief specifies the content in-memory of a row in the frontend view metadata view
 *
 */

struct FrontendViewDescriptor {
  int32_t viewId;       /**< viewId starts at 0 for valid views. */
  std::string viewName; /**< viewName is the name of the view view -must be unique */
  std::string viewState;
  std::string imageHash;
  std::string updateTime;
  std::string viewMetadata;
  int32_t userId;
};

#endif  // FRONTEND_VIEW_DESCRIPTOR
