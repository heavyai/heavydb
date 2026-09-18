/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/JSONSchemas.h"

namespace gfx {

namespace {
static const std::string g_manifest_schema = R"manifest(
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "$id": "http://omnisci.com/shader_manifest_schema.json",
  "title": "Shader Manifest",
  "description": "",
  "type": "object",
  "properties": {
    "sources": {
      "type": "array",
      "items": { "$ref": "#/definitions/source" }
    }
  },
  "definitions": {
    "source": {
      "type": "object",
      "required": [ "filename", "file_path", "internal_path", "class", "language" ],
      "properties": {
        "filename": {
          "type": "string",
          "description": "Name of the template or shader file."
        },
        "file_path": {
          "type": "string",
          "description": "System path to the file relative to the manifests location."
        },
        "internal_path": {
          "type": "string",
          "description": "Path to use for lookups in the Library. Acts as simple namespacing."
        },
        "class": {
          "type": "string",
          "description": "Type of shader, typically a shader stage.",
          "enum": [ "vertex", "fragment", "geometry", "tess_control", "tess_eval", "compute", "ray_gen", "ray_closest_hit", "ray_miss", "ray_intersection", "ray_callable", "mesh", "task", "glsl" ]
        },
        "language": {
          "type": "string",
          "description": "Shader language.",
          "enum": [ "glsl" ]
        }
      }
    }
  },
  "required": [
    "sources"
  ]
}
)manifest";

static const std::string g_builder_schema = R"builder(
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "$id": "http://omnisci.com/shader_builder_schema.json",
  "title": "Shader Builder Serialization",
  "description": "",
  "type": "object",
  "properties": {
    "baseTemplate": {
      "description": "",
      "type": "string"
    },
    "entryPoint": {
      "description": "Optional name of function to use as the entry point. Defaults to main",
      "type": "string"
    },
    "operators": {
      "type": "array",
      "items": { "$ref": "#/definitions/operator" }
    },
    "subroutines": {
      "type": "array",
      "items": { "$ref": "#/definitions/subroutine" }
    }
  },
  "definitions": {
    "operator": {
      "type": "object",
      "required": [ "name", "string1", "string2", "required" ],
      "properties": {
        "name": {
          "type": "string",
          "description": "The name of the shader template."
        },
        "string1": {
          "type": "string",
          "description": "Operator parameter."
        },
        "string2": {
          "type": "string",
          "description": "Operator parameter."
        },
        "sub_builder": {
          "type": "object",
          "description": "Optional embedded sub-Builder parameter"
        },
        "required": {
          "type": "boolean",
          "description": "Is operator success a requirement."
        }
      }
    },
    "subroutine": {
      "type": "object",
      "required": [ "call", "target", "required" ],
      "properties": {
        "call": {
          "type": "string",
          "description": "Function call to replace."
        },
        "target": {
          "type": "string",
          "description": "New function to use."
        },
        "required": {
          "type": "boolean",
          "description": "Is replacement required."
        }
      }
    }
  },
  "required": [
    "baseTemplate"
  ]
}
)builder";
}  // namespace

const std::string& get_shader_manifest_schema() {
  return g_manifest_schema;
}

const std::string& get_shader_builder_schema() {
  return g_builder_schema;
}

}  // namespace gfx
