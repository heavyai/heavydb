{
  "baseTemplate": "Marks/lineTemplate.frag",
  "operators": [
    {
      "name": "kReplaceFirstTag",
      "string1": "FragmentShaderInputs",
      "string2": "FragmentShaderInputs {\n  layout (location = 0) flat uint64_t fRowId;\n  layout (location = 1) vec2 fNormDistCoords;\n  layout (location = 2) flat vec4 fColor;\n  layout (location = 3) flat float fOpacity;\n  layout (location = 4) flat float fStrokeOpacity;\n}",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "RenderPropertyTypeInfos",
      "string2": "#define inTx double\n#define inTxEnum DOUBLE\n#define outTx float\n#define outTxEnum FLOAT\n\n#define inTy double\n#define inTyEnum DOUBLE\n#define outTy float\n#define outTyEnum FLOAT\n\n#define inTstrokeOpacity float\n#define inTstrokeOpacityEnum FLOAT\n#define outTstrokeOpacity float\n#define outTstrokeOpacityEnum FLOAT\n\n#define inTstrokeWidth float\n#define inTstrokeWidthEnum FLOAT\n#define outTstrokeWidth float\n#define outTstrokeWidthEnum FLOAT\n\n#define inTopacity float\n#define inTopacityEnum FLOAT\n#define outTopacity float\n#define outTopacityEnum FLOAT\n\n#define inTstrokeColor int64_t\n#define inTstrokeColorEnum INT64_ARB\n#define outTstrokeColor vec4\n#define outTstrokeColorEnum FLOAT_VEC4\n\n#define inTlineJoin int\n#define inTlineJoinEnum INT\n#define outTlineJoin int\n#define outTlineJoinEnum INT\n\n#define inTmiterLimit float\n#define inTmiterLimitEnum FLOAT\n#define outTmiterLimit float\n#define outTmiterLimitEnum FLOAT\n\n#define useUmiterLimit 1\n#define useUlineJoin 1\n#define useUopacity 1\n#define useUstrokeWidth 1\n#define useUstrokeOpacity 1\n",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "useSSBO",
      "string2": "1",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "lineData",
      "string2": "struct LineDataType {\n  int64_t color;\n  int64_t size;\n  int64_t rowid;\n};\nlayout(std430) readonly buffer LineData {\n  LineDataType lineData[];\n};",
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "lineData[iSSBOIndex].<id>",
      "string2": "lineData[iSSBOIndex].rowid",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "numid",
      "string2": "1",
      "required": false
    },
    {
      "name": "kReplaceWithSubBuilder",
      "string1": "getx",
      "string2": "",
      "sub_builder": {
        "baseTemplate": "Scales/quantitativeScaleTemplate.vert",
        "operators": [
          {
            "name": "kReplaceAllMultiple",
            "string1": "",
            "string2": "",
            "required": false,
            "name_map": [
              {
                "string1": "<domainType>",
                "string2": "double"
              },
              {
                "string1": "<domainTypeEnum>",
                "string2": "DOUBLE"
              },
              {
                "string1": "<rangeType>",
                "string2": "float"
              },
              {
                "string1": "<rangeTypeEnum>",
                "string2": "FLOAT"
              },
              {
                "string1": "<numDomains>",
                "string2": "2"
              },
              {
                "string1": "<numRanges>",
                "string2": "2"
              },
              {
                "string1": "<doAccum>",
                "string2": "0"
              },
              {
                "string1": "<name>",
                "string2": "x_x"
              }
            ]
          },
          {
            "name": "kReplaceAllTags",
            "string1": "useClamp",
            "string2": "0",
            "required": false
          }
        ]
      },
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "getx(projectx(x))",
      "string2": "evalQuantitativeScale_x_x(x)",
      "required": false
    },
    {
      "name": "kReplaceWithSubBuilder",
      "string1": "gety",
      "string2": "",
      "sub_builder": {
        "baseTemplate": "Scales/quantitativeScaleTemplate.vert",
        "operators": [
          {
            "name": "kReplaceAllMultiple",
            "string1": "",
            "string2": "",
            "required": false,
            "name_map": [
              {
                "string1": "<domainType>",
                "string2": "double"
              },
              {
                "string1": "<domainTypeEnum>",
                "string2": "DOUBLE"
              },
              {
                "string1": "<rangeType>",
                "string2": "float"
              },
              {
                "string1": "<rangeTypeEnum>",
                "string2": "FLOAT"
              },
              {
                "string1": "<numDomains>",
                "string2": "2"
              },
              {
                "string1": "<numRanges>",
                "string2": "2"
              },
              {
                "string1": "<doAccum>",
                "string2": "0"
              },
              {
                "string1": "<name>",
                "string2": "y_y"
              }
            ]
          },
          {
            "name": "kReplaceAllTags",
            "string1": "useClamp",
            "string2": "0",
            "required": false
          }
        ]
      },
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "gety(projecty(y))",
      "string2": "evalQuantitativeScale_y_y(y)",
      "required": false
    },
    {
      "name": "kReplaceWithSubBuilder",
      "string1": "getstrokeColor",
      "string2": "",
      "sub_builder": {
        "baseTemplate": "Scales/ordinalScaleTemplate.vert",
        "operators": [
          {
            "name": "kReplaceAllMultiple",
            "string1": "",
            "string2": "",
            "required": false,
            "name_map": [
              {
                "string1": "<domainType>",
                "string2": "int64_t"
              },
              {
                "string1": "<domainTypeEnum>",
                "string2": "INT64_ARB"
              },
              {
                "string1": "<rangeType>",
                "string2": "vec4"
              },
              {
                "string1": "<rangeTypeEnum>",
                "string2": "FLOAT_VEC4"
              },
              {
                "string1": "<numDomains>",
                "string2": "20"
              },
              {
                "string1": "<numRanges>",
                "string2": "20"
              },
              {
                "string1": "<doAccum>",
                "string2": "0"
              },
              {
                "string1": "<name>",
                "string2": "lines_strokeColor_strokeColor"
              }
            ]
          }
        ]
      },
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "getstrokeColor(lineData[iSSBOIndex].<strokeColor>)",
      "string2": "evalOrdinalScale_lines_strokeColor_strokeColor(domainType_lines_strokeColor_strokeColor(lineData[iSSBOIndex].color))",
      "required": false
    },
    {
      "name": "kAppendTemplate",
      "string1": "Outputs/output_ID.glsl",
      "string2": "",
      "required": false
    },
    {
      "name": "kAppendTemplate",
      "string1": "Marks/mainTemplate_StdRaster.glsl",
      "string2": "",
      "required": false
    }
  ],
  "subroutines": [
    {
      "call": "transformstrokeColorToRGB",
      "target": "transformRGBtoRGB",
      "required": true
    }
  ]
}