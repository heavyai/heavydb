/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/ColorRGBA.h"

#include <algorithm>

#include "GfxDriver/Colors/ColorInitializer.h"
#include "GfxDriver/Colors/ColorUnion.h"
#include "Logger/Logger.h"

namespace gfx {

// const details::ConvertToRGBAChannel ColorRGBA::rgbaChannelConverter =
// details::ConvertToRGBAChannel();

static const std::unordered_map<std::string, std::string> colorKeywords = {
    {"aliceblue", "#F0F8FF"},
    {"antiquewhite", "#FAEBD7"},
    {"aqua", "#00FFFF"},
    {"aquamarine", "#7FFFD4"},
    {"azure", "#F0FFFF"},
    {"beige", "#F5F5DC"},
    {"bisque", "#FFE4C4"},
    {"black", "#000000"},
    {"blanchedalmond", "#FFEBCD"},
    {"blue", "#0000FF"},
    {"blueviolet", "#8A2BE2"},
    {"brown", "#A52A2A"},
    {"burlywood", "#DEB887"},
    {"cadetblue", "#5F9EA0"},
    {"chartreuse", "#7FFF00"},
    {"chocolate", "#D2691E"},
    {"coral", "#FF7F50"},
    {"cornflowerblue", "#6495ED"},
    {"cornsilk", "#FFF8DC"},
    {"crimson", "#DC143C"},
    {"cyan", "#00FFFF"},
    {"darkblue", "#00008B"},
    {"darkcyan", "#008B8B"},
    {"darkgoldenrod", "#B8860B"},
    {"darkgray", "#A9A9A9"},
    {"darkgreen", "#006400"},
    {"darkkhaki", "#BDB76B"},
    {"darkmagenta", "#8B008B"},
    {"darkolivegreen", "#556B2F"},
    {"darkorange", "#FF8C00"},
    {"darkorchid", "#9932CC"},
    {"darkred", "#8B0000"},
    {"darksalmon", "#E9967A"},
    {"darkseagreen", "#8FBC8F"},
    {"darkslateblue", "#483D8B"},
    {"darkslategray", "#2F4F4F"},
    {"darkturquoise", "#00CED1"},
    {"darkviolet", "#9400D3"},
    {"deeppink", "#FF1493"},
    {"deepskyblue", "#00BFFF"},
    {"dimgray", "#696969"},
    {"dodgerblue", "#1E90FF"},
    {"firebrick", "#B22222"},
    {"floralwhite", "#FFFAF0"},
    {"forestgreen", "#228B22"},
    {"fuchsia", "#FF00FF"},
    {"gainsboro", "#DCDCDC"},
    {"ghostwhite", "#F8F8FF"},
    {"gold", "#FFD700"},
    {"goldenrod", "#DAA520"},
    {"gray", "#808080"},
    {"green", "#008000"},
    {"greenyellow", "#ADFF2F"},
    {"honeydew", "#F0FFF0"},
    {"hotpink", "#FF69B4"},
    {"indianred", "#CD5C5C"},
    {"indigo", "#4B0082"},
    {"ivory", "#FFFFF0"},
    {"khaki", "#F0E68C"},
    {"lavender", "#E6E6FA"},
    {"lavenderblush", "#FFF0F5"},
    {"lawngreen", "#7CFC00"},
    {"lemonchiffon", "#FFFACD"},
    {"lightblue", "#ADD8E6"},
    {"lightcoral", "#F08080"},
    {"lightcyan", "#E0FFFF"},
    {"lightgoldenrodyellow", "#FAFAD2"},
    {"lightgray", "#D3D3D3"},
    {"lightgreen", "#90EE90"},
    {"lightpink", "#FFB6C1"},
    {"lightsalmon", "#FFA07A"},
    {"lightseagreen", "#20B2AA"},
    {"lightskyblue", "#87CEFA"},
    {"lightslategray", "#778899"},
    {"lightsteelblue", "#B0C4DE"},
    {"lightyellow", "#FFFFE0"},
    {"lime", "#00FF00"},
    {"limegreen", "#32CD32"},
    {"linen", "#FAF0E6"},
    {"magenta", "#FF00FF"},
    {"maroon", "#800000"},
    {"mediumaquamarine", "#66CDAA"},
    {"mediumblue", "#0000CD"},
    {"mediumorchid", "#BA55D3"},
    {"mediumpurple", "#9370DB"},
    {"mediumseagreen", "#3CB371"},
    {"mediumslateblue", "#7B68EE"},
    {"mediumspringgreen", "#00FA9A"},
    {"mediumturquoise", "#48D1CC"},
    {"mediumvioletred", "#C71585"},
    {"midnightblue", "#191970"},
    {"mintcream", "#F5FFFA"},
    {"mistyrose", "#FFE4E1"},
    {"moccasin", "#FFE4B5"},
    {"navajowhite", "#FFDEAD"},
    {"navy", "#000080"},
    {"oldlace", "#FDF5E6"},
    {"olive", "#808000"},
    {"olivedrab", "#6B8E23"},
    {"orange", "#FFA500"},
    {"orangered", "#FF4500"},
    {"orchid", "#DA70D6"},
    {"palegoldenrod", "#EEE8AA"},
    {"palegreen", "#98FB98"},
    {"paleturquoise", "#AFEEEE"},
    {"palevioletred", "#DB7093"},
    {"papayawhip", "#FFEFD5"},
    {"peachpuff", "#FFDAB9"},
    {"peru", "#CD853F"},
    {"pink", "#FFC0CB"},
    {"plum", "#DDA0DD"},
    {"powderblue", "#B0E0E6"},
    {"purple", "#800080"},
    {"rebeccapurple", "#663399"},
    {"red", "#FF0000"},
    {"rosybrown", "#BC8F8F"},
    {"royalblue", "#4169E1"},
    {"saddlebrown", "#8B4513"},
    {"salmon", "#FA8072"},
    {"sandybrown", "#F4A460"},
    {"seagreen", "#2E8B57"},
    {"seashell", "#FFF5EE"},
    {"sienna", "#A0522D"},
    {"silver", "#C0C0C0"},
    {"skyblue", "#87CEEB"},
    {"slateblue", "#6A5ACD"},
    {"slategray", "#708090"},
    {"snow", "#FFFAFA"},
    {"springgreen", "#00FF7F"},
    {"steelblue", "#4682B4"},
    {"tan", "#D2B48C"},
    {"teal", "#008080"},
    {"thistle", "#D8BFD8"},
    {"tomato", "#FF6347"},
    {"turquoise", "#40E0D0"},
    {"violet", "#EE82EE"},
    {"wheat", "#F5DEB3"},
    {"white", "#FFFFFF"},
    {"whitesmoke", "#F5F5F5"},
    {"yellow", "#FFFF00"},
    {"yellowgreen", "#9ACD32"}};

const std::array<std::string, 2> ColorRGBA::funcNames = {"rgb", "rgba"};
const std::array<std::string, 2> ColorRGBA::argRegexStrs = {
    "\\s*(\\d{1,3})\\s*,\\s*(\\d{1,3})\\s*,\\s*(\\d{1,3})\\s*",
    "\\s*(\\d{1,3})\\s*,\\s*(\\d{1,3})\\s*,\\s*(\\d{1,3})\\s*,\\s*([0,1](?:\\.\\d*)?)"
    "\\s*"};

const PackedFloatColorConverters::ConvertUInt8To0to1 ColorRGBA::convertRGB =
    PackedFloatColorConverters::ConvertUInt8To0to1();
const ColorValidators::Clamp0to1f ColorRGBA::clampRGB = ColorValidators::Clamp0to1f();

static const std::regex hexRegex(
    "^#([0-9,a-f,A-F]{2})([0-9,a-f,A-F]{2})([0-9,a-f,A-F]{2})\\s*$",
    std::regex_constants::icase);

static bool initFromHexString(ColorRGBA& colorObj, const std::string& colorStr) {
  uint8_t r, g, b;
  std::smatch matches;

  if (std::regex_match(colorStr, matches, hexRegex)) {
    r = static_cast<uint8_t>(std::min(std::stoul(matches[1], nullptr, 16), 255ul));
    g = static_cast<uint8_t>(std::min(std::stoul(matches[2], nullptr, 16), 255ul));
    b = static_cast<uint8_t>(std::min(std::stoul(matches[3], nullptr, 16), 255ul));

    colorObj.setFromPackedComponents(r, g, b);
    return true;
  }
  return false;
}

static decltype(colorKeywords)::const_iterator getColorKeywordItr(
    const std::string& colorStr) {
  std::string lowerColorStr = colorStr;
  std::transform(
      lowerColorStr.begin(), lowerColorStr.end(), lowerColorStr.begin(), ::tolower);
  return colorKeywords.find(lowerColorStr);
}

ColorRGBA& ColorRGBA::operator=(const ColorUnion& rhs) {
  return operator=(rhs.get<ColorRGBA>());
}

void ColorRGBA::initFromCSSString(const std::string& colorStr) {
  try {
    ColorInitializer::initColorFromColorString(*this, colorStr);
  } catch (RenderError& err) {
    // TODO(croot): indicate why
    THROW_RUNTIME_EX("\"" + colorStr + "\" is not a valid ColorRGBA string.");
  }
}

ColorRGBA::operator std::string() const {
  std::ostringstream s;
  auto converter = PackedFloatColorConverters::ConvertUInt8To0to1();
  int i = 0;
  s << "rgba(";
  for (i = 0; i < static_cast<int>(_colorArray.size()) - 1; ++i) {
    s << static_cast<int>(converter.inverse(_colorArray[i])) << ",";
  }
  s << _colorArray[i];

  s << ")";
  return s.str();
}

bool ColorRGBA::isColorString(const std::string& colorStr) {
  static const std::regex rgbRegex(
      "^" + funcNames[0] + "\\s*\\(" + argRegexStrs[0] + "\\)\\s*$",
      std::regex_constants::icase);
  static const std::regex rgbaRegex(
      "^" + funcNames[1] + "\\s*\\(" + argRegexStrs[1] + "\\)\\s*$",
      std::regex_constants::icase);

  return (std::regex_match(colorStr, rgbRegex) || std::regex_match(colorStr, rgbaRegex) ||
          std::regex_match(colorStr, hexRegex) ||
          getColorKeywordItr(colorStr) != colorKeywords.end());
}

bool ColorRGBA::getNonFuncMatch(const std::string& colorStr, std::smatch& nonFuncMatch) {
  if (std::regex_match(colorStr, nonFuncMatch, hexRegex)) {
    return true;
  }

  return getColorKeywordItr(colorStr) != colorKeywords.end();
}

void ColorRGBA::initFromFuncArgs(const std::smatch& argMatch) {
  uint8_t r, g, b;

  CHECK(argMatch.size() == 4 || argMatch.size() == 5);

  r = static_cast<uint8_t>(std::min(std::stoul(argMatch.str(1)), 255ul));
  g = static_cast<uint8_t>(std::min(std::stoul(argMatch.str(2)), 255ul));
  b = static_cast<uint8_t>(std::min(std::stoul(argMatch.str(3)), 255ul));

  setFromPackedComponents(r, g, b);

  if (argMatch.size() == 5) {
    _colorArray[3] = Math::clamp(std::stof(argMatch.str(4)), 0.0f, 1.0f);
  }
}

void ColorRGBA::initFromNonFunc(const std::string& colorStr,
                                const std::smatch& nonFuncMatch) {
  if (initFromHexString(*this, colorStr)) {
    return;
  }

  auto itr = getColorKeywordItr(colorStr);
  if (itr != colorKeywords.end()) {
    if (initFromHexString(*this, itr->second)) {
      return;
    }
  }

  // TODO(croot): Give a more detailed reason as to why
  THROW_RUNTIME_EX("Cannot initialize a ColorRGBA object from the string \"" + colorStr +
                   "\"");
}

}  // namespace gfx
