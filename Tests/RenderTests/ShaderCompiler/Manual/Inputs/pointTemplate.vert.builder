{
  "baseTemplate": "Marks/pointTemplate.vert",
  "operators": [
    {
      "name": "kReplaceFirstTag",
      "string1": "FragmentShaderInputs",
      "string2": "FragmentShaderInputs {\n  layout (location = 0) flat uint64_t fRowId;\n  layout (location = 1) flat vec4 fColor;\n  layout (location = 2) flat float fPointSize;\n}",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "useKey",
      "string2": "1",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "VertexProperties",
      "string2": "layout (location = 0) in int64_t key;\nlayout (location = 1) in double x;\nlayout (location = 2) in double y;\nlayout (location = 3) in int64_t id;\n",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "UniformProperties",
      "string2": "POINT_VERT_UBO_TYPE {\n  mat3x2 uViewProjMatrix;\n  uint64_t invalidKey;\n  uint propCompressionBits;\n  float size;\n  float fillOpacity;\n  vec4 fillColor;\n  float opacity;\n}",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "PropertyGetters",
      "string2": "int64_t getkey(int64_t key) {\n  return key;\n}\n\nint64_t getid(int64_t id) {\n  return id;\n}\n\nfloat gety(double y) {\n  return float(y);\n}\n\nfloat getx(double x) {\n  return float(x);\n}\n\nfloat getsize(float size) {\n  return size;\n}\n\nfloat getfillOpacity(float fillOpacity) {\n  return fillOpacity;\n}\n\nvec4 getfillColor(vec4 fillColor) {\n  return fillColor;\n}\n\nfloat getopacity(float opacity) {\n  return opacity;\n}\n\n",
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
    }
  ],
  "subroutines": [
    {
      "call": "isNullValFunc_x_x",
      "target": "isNullValPassThru_x_x",
      "required": false
    },
    {
      "call": "quantTransform_x_x",
      "target": "passThruTransform_x_x",
      "required": true
    },
    {
      "call": "isNullValFunc_y_y",
      "target": "isNullValPassThru_y_y",
      "required": false
    },
    {
      "call": "transformfillColorToRGB",
      "target": "transformRGBtoRGB",
      "required": true
    },
    {
      "call": "quantInterp_x_x",
      "target": "defaultInterp_x_x",
      "required": true
    },
    {
      "call": "quantInterp_y_y",
      "target": "defaultInterp_y_y",
      "required": true
    },
    {
      "call": "quantTransform_y_y",
      "target": "passThruTransform_y_y",
      "required": true
    }
  ]
}