{
  "baseTemplate": "Marks/fastSymbolTemplate.frag",
  "operators": [
    {
      "name": "kReplaceFirstTag",
      "string1": "FragmentShaderInputs",
      "string2": "FragmentShaderInputs {\n  layout (location = 0) flat uint64_t fRowId;\n  layout (location = 1) flat uint fShapeType;\n  layout (location = 2) flat vec4 fSymbolScale;\n  layout (location = 3) flat float fPointSize;\n  layout (location = 4) flat vec4 fFillColor;\n  layout (location = 5) flat vec4 fStrokeColor;\n  layout (location = 6) flat float fStrokeWidth;\n  layout (location = 7) flat float fApproxMaxCoverage;\n}",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "numSymbolVerts",
      "string2": "53",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "isMultiSampling",
      "string2": "1",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "usingGeomOrMeshShader",
      "string2": "0",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "RenderPropertyTypeInfos",
      "string2": "#define inTshape int64_t\n#define inTshapeEnum INT64_ARB\n#define outTshape uint\n#define outTshapeEnum UNSIGNED_INT\n\n#define inTheight int64_t\n#define inTheightEnum INT64_ARB\n#define outTheight int\n#define outTheightEnum INT\n\n#define inTwidth int64_t\n#define inTwidthEnum INT64_ARB\n#define outTwidth int\n#define outTwidthEnum INT\n\n#define inTopacity float\n#define inTopacityEnum FLOAT\n#define outTopacity float\n#define outTopacityEnum FLOAT\n\n#define inTstrokeColor vec4\n#define inTstrokeColorEnum FLOAT_VEC4\n#define outTstrokeColor vec4\n#define outTstrokeColorEnum FLOAT_VEC4\n\n#define inTx double\n#define inTxEnum DOUBLE\n#define outTx float\n#define outTxEnum FLOAT\n\n#define inTfillColor int64_t\n#define inTfillColorEnum INT64_ARB\n#define outTfillColor vec4\n#define outTfillColorEnum FLOAT_VEC4\n\n#define inTstrokeWidth float\n#define inTstrokeWidthEnum FLOAT\n#define outTstrokeWidth float\n#define outTstrokeWidthEnum FLOAT\n\n#define inTstrokeOpacity float\n#define inTstrokeOpacityEnum FLOAT\n#define outTstrokeOpacity float\n#define outTstrokeOpacityEnum FLOAT\n\n#define inTy double\n#define inTyEnum DOUBLE\n#define outTy float\n#define outTyEnum FLOAT\n\n#define inTx2 float\n#define inTx2Enum FLOAT\n#define outTx2 float\n#define outTx2Enum FLOAT\n\n#define inTyc float\n#define inTycEnum FLOAT\n#define outTyc float\n#define outTycEnum FLOAT\n\n#define inTxc float\n#define inTxcEnum FLOAT\n#define outTxc float\n#define outTxcEnum FLOAT\n\n#define inTfillOpacity float\n#define inTfillOpacityEnum FLOAT\n#define outTfillOpacity float\n#define outTfillOpacityEnum FLOAT\n\n#define inTy2 float\n#define inTy2Enum FLOAT\n#define outTy2 float\n#define outTy2Enum FLOAT\n\n#define useUy2 1\n#define useUfillOpacity 1\n#define useUxc 1\n#define useUyc 1\n#define useUx2 1\n#define useUstrokeWidth 1\n#define useUstrokeOpacity 1\n#define useUstrokeColor 1\n#define useUopacity 1\n",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "useSSBO",
      "string2": "0",
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
      "string1": "getshape",
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
                "string2": "int"
              },
              {
                "string1": "<rangeTypeEnum>",
                "string2": "INT"
              },
              {
                "string1": "<numDomains>",
                "string2": "3"
              },
              {
                "string1": "<numRanges>",
                "string2": "3"
              },
              {
                "string1": "<doAccum>",
                "string2": "0"
              },
              {
                "string1": "<name>",
                "string2": "symbol_shape"
              }
            ]
          }
        ]
      },
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "getshape(shape)",
      "string2": "evalOrdinalScale_symbol_shape(domainType_symbol_shape(shape))",
      "required": false
    },
    {
      "name": "kReplaceWithSubBuilder",
      "string1": "getheight",
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
                "string2": "float"
              },
              {
                "string1": "<rangeTypeEnum>",
                "string2": "FLOAT"
              },
              {
                "string1": "<numDomains>",
                "string2": "3"
              },
              {
                "string1": "<numRanges>",
                "string2": "3"
              },
              {
                "string1": "<doAccum>",
                "string2": "0"
              },
              {
                "string1": "<name>",
                "string2": "symbol_size_height"
              }
            ]
          }
        ]
      },
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "getheight(height)",
      "string2": "evalOrdinalScale_symbol_size_height(domainType_symbol_size_height(height))",
      "required": false
    },
    {
      "name": "kReplaceWithSubBuilder",
      "string1": "getwidth",
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
                "string2": "float"
              },
              {
                "string1": "<rangeTypeEnum>",
                "string2": "FLOAT"
              },
              {
                "string1": "<numDomains>",
                "string2": "3"
              },
              {
                "string1": "<numRanges>",
                "string2": "3"
              },
              {
                "string1": "<doAccum>",
                "string2": "0"
              },
              {
                "string1": "<name>",
                "string2": "symbol_size_width"
              }
            ]
          }
        ]
      },
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "getwidth(width)",
      "string2": "evalOrdinalScale_symbol_size_width(domainType_symbol_size_width(width))",
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
      "string1": "getfillColor",
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
                "string2": "color_fillColor"
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
      "string1": "getfillColor(fillColor)",
      "string2": "evalQuantitativeScale_color_fillColor(domainType_color_fillColor(fillColor))",
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
      "name": "kReplaceAll",
      "string1": "projectx2(x2)",
      "string2": "x2",
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "projectyc(yc)",
      "string2": "yc",
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "projectxc(xc)",
      "string2": "xc",
      "required": false
    },
    {
      "name": "kReplaceAll",
      "string1": "projecty2(y2)",
      "string2": "y2",
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
      "call": "mapDistanceToColor",
      "target": "mapDistanceToColorFillAndStroke",
      "required": true
    }
  ]
}