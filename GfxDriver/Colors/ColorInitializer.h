/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Colors/ColorHCL.h"
#include "GfxDriver/Colors/ColorHSL.h"
#include "GfxDriver/Colors/ColorLAB.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Colors/Utils.h"

namespace gfx {

struct ColorInitializer {
  template <typename T>
  static void initColorFromColorString(T& colorObj, const std::string& str) {
    std::smatch funcMatch, nonFuncMatch;
    if (getFuncMatch<T>(str, funcMatch) || T::getNonFuncMatch(str, nonFuncMatch)) {
      initColorFromColorStringRecursive(colorObj, str, funcMatch, nonFuncMatch);
    } else {
      THROW_RUNTIME_EX("\"" + str + "\" is not a valid color string.");
    }
  }

  static ColorType getColorTypeFromString(const std::string& colorStr,
                                          const bool checkFuncs = true,
                                          const bool checkNonFuncs = true) {
    std::smatch match1, match2;
    return getColorTypeFromStringInternal(
        colorStr, (checkFuncs ? &match1 : nullptr), (checkNonFuncs ? &match2 : nullptr));
  }

  static bool isColorString(const std::string& colorStr,
                            const bool checkFuncs = true,
                            const bool checkNonFuncs = true) {
    std::smatch match1, match2;
    try {
      getColorTypeFromStringInternal(colorStr,
                                     (checkFuncs ? &match1 : nullptr),
                                     (checkNonFuncs ? &match2 : nullptr),
                                     false);
    } catch (::gfx::RenderError& err) {
      return false;
    }

    return true;
  }

  template <typename T>
  static bool getFuncMatch(const std::string& colorStr,
                           std::smatch& funcMatch,
                           bool addParens = true) {
    std::string funcRegex = "^\\s*(";
    int idx = 0;
    for (; idx < static_cast<int>(T::funcNames.size()) - 1; idx++) {
      funcRegex += T::funcNames[idx] + "|";
    }
    funcRegex += T::funcNames[idx] + ")\\s*" + (addParens ? "\\((.*)\\)\\s*" : "") + "$";
    return std::regex_match(
        colorStr, funcMatch, std::regex(funcRegex, std::regex_constants::icase));
  }

 private:
  static ColorType getColorTypeFromStringInternal(const std::string& colorStr,
                                                  std::smatch* funcMatch = nullptr,
                                                  std::smatch* nonFuncMatch = nullptr,
                                                  const bool log = true) {
    if ((funcMatch && getFuncMatch<ColorRGBA>(colorStr, *funcMatch)) ||
        (nonFuncMatch && ColorRGBA::getNonFuncMatch(colorStr, *nonFuncMatch))) {
      return ColorType::RGBA;
    } else if ((funcMatch && getFuncMatch<ColorHSL>(colorStr, *funcMatch)) ||
               (nonFuncMatch && ColorHSL::getNonFuncMatch(colorStr, *nonFuncMatch))) {
      return ColorType::HSL;
    } else if ((funcMatch && getFuncMatch<ColorLAB>(colorStr, *funcMatch)) ||
               (nonFuncMatch && ColorLAB::getNonFuncMatch(colorStr, *nonFuncMatch))) {
      return ColorType::LAB;
    } else if ((funcMatch && getFuncMatch<ColorHCL>(colorStr, *funcMatch)) ||
               (nonFuncMatch && ColorHCL::getNonFuncMatch(colorStr, *nonFuncMatch))) {
      return ColorType::HCL;
    } else {
      // TODO(croot): indicate why
      const auto errStr = "\"" + colorStr + "\" is not a valid color string.";
      if (log) {
        THROW_RUNTIME_EX(errStr);
      } else {
        THROW_RUNTIME_EX_NO_LOG(errStr);
      }
    }
  }

  template <typename T>
  static void initColorFromColorStringRecursive(T& colorObj,
                                                const std::string& str,
                                                std::smatch& funcMatch,
                                                std::smatch& nonFuncMatch) {
    if (!funcMatch.empty()) {
      std::smatch argMatch;
      std::string args = funcMatch.str(2);
      if (getArgMatch<T>(args, funcMatch, argMatch)) {
        colorObj.initFromFuncArgs(argMatch);
        return;
      } else {
        auto colorType = getColorTypeFromStringInternal(args, &funcMatch, &nonFuncMatch);
        switch (colorType) {
          case ColorType::RGBA: {
            ColorRGBA rgba;
            initColorFromColorStringRecursive(rgba, args, funcMatch, nonFuncMatch);
            convertColor(rgba, colorObj);
            break;
          }
          case ColorType::HSL: {
            ColorHSL hsl;
            initColorFromColorStringRecursive(hsl, args, funcMatch, nonFuncMatch);
            convertColor(hsl, colorObj);
            break;
          }
          case ColorType::LAB: {
            ColorLAB lab;
            initColorFromColorStringRecursive(lab, args, funcMatch, nonFuncMatch);
            convertColor(lab, colorObj);
            break;
          }
          case ColorType::HCL: {
            ColorHCL hcl;
            initColorFromColorStringRecursive(hcl, args, funcMatch, nonFuncMatch);
            convertColor(hcl, colorObj);
            break;
          }
          default:
            THROW_RUNTIME_EX("Color type " + std::to_string(static_cast<int>(colorType)) +
                             " is not supported to init from a color string.");
        }
      }
    } else {
      colorObj.initFromNonFunc(str, nonFuncMatch);
    }
  }

  template <typename T>
  static bool getArgMatch(const std::string& args,
                          const std::smatch& funcMatch,
                          std::smatch& argMatch) {
    int idx = 0;
    for (auto& funcName : T::funcNames) {
      if (std::regex_match(funcMatch.str(1),
                           std::regex(funcName, std::regex_constants::icase))) {
        return std::regex_match(
            args,
            argMatch,
            std::regex("^" + T::argRegexStrs[idx] + "$", std::regex_constants::icase));
      }
      idx++;
    }

    return false;
  }
};

}  // namespace gfx
