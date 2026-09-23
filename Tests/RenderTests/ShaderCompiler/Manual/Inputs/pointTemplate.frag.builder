{
  "baseTemplate": "Marks/pointTemplate.frag",
  "operators": [
    {
      "name": "kReplaceFirstTag",
      "string1": "FragmentShaderInputs",
      "string2": "FragmentShaderInputs {\n  layout (location = 0) flat uint64_t fRowId;\n  layout (location = 1) flat vec4 fColor;\n  layout (location = 2) flat float fPointSize;\n}",
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
      "string1": "doDeviceLostTestLoop",
      "string2": "0",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "RenderPropertyTypeInfos",
      "string2": "#define inTx double\n#define inTxEnum DOUBLE\n#define outTx float\n#define outTxEnum FLOAT\n\n#define inTy double\n#define inTyEnum DOUBLE\n#define outTy float\n#define outTyEnum FLOAT\n\n#define inTopacity float\n#define inTopacityEnum FLOAT\n#define outTopacity float\n#define outTopacityEnum FLOAT\n\n#define inTfillColor vec4\n#define inTfillColorEnum FLOAT_VEC4\n#define outTfillColor vec4\n#define outTfillColorEnum FLOAT_VEC4\n\n#define inTfillOpacity float\n#define inTfillOpacityEnum FLOAT\n#define outTfillOpacity float\n#define outTfillOpacityEnum FLOAT\n\n#define inTsize float\n#define inTsizeEnum FLOAT\n#define outTsize float\n#define outTsizeEnum FLOAT\n\n#define useUsize 1\n#define useUfillOpacity 1\n#define useUfillColor 1\n#define useUopacity 1\n",
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
  ]
}