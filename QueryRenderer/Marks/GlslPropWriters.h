/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

#include "QueryRenderer/Marks/BaseRenderProperty.h"

namespace QueryRenderer {

class GlslAbstractPropWriter {
 public:
  GlslAbstractPropWriter(const BaseRenderProperty* prop)
      : prop_{prop}, parent_writer_{nullptr} {}

  GlslAbstractPropWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer)
      : prop_(parent_writer->prop_), parent_writer_(std::move(parent_writer)) {}

  virtual ~GlslAbstractPropWriter() {}
  virtual std::string getTemplateStr() const = 0;
  virtual std::string getFinalStr() const = 0;

 protected:
  const BaseRenderProperty* prop_;
  const std::unique_ptr<GlslAbstractPropWriter> parent_writer_;
};

class PropArgWriter : public GlslAbstractPropWriter {
 public:
  PropArgWriter(const BaseRenderProperty* prop, const std::string* ssbo_instance_name)
      : GlslAbstractPropWriter(prop), ssbo_instance_name_{ssbo_instance_name} {
    CHECK(!ssbo_instance_name || prop->getDataColumnNameRef().length() > 0)
        << "prop " << prop->getName() << " isn't initialized with buffer data";
  }

  std::string getTemplateStr() const final {
    return (ssbo_instance_name_
                ? *ssbo_instance_name_ + "[iSSBOIndex].<" + prop_->getName() + ">"
                : prop_->getName());
  }

  std::string getFinalStr() const final {
    return (ssbo_instance_name_
                ? *ssbo_instance_name_ + "[iSSBOIndex]." + prop_->getDataColumnName()
                : prop_->getName());
  }

 protected:
  const std::string* ssbo_instance_name_;
};

class DecimalWriter : public GlslAbstractPropWriter {
 public:
  DecimalWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer)
      : GlslAbstractPropWriter(std::move(parent_writer)) {}

  std::string getTemplateStr() const final { return parent_writer_->getTemplateStr(); }

  std::string getFinalStr() const final {
    return "convertDecimalToDouble(" + parent_writer_->getFinalStr() + ", " +
           prop_->getName() + "_ExpScale)";
  }
};

class DecompressGeoCoordWriter : public GlslAbstractPropWriter {
 public:
  DecompressGeoCoordWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer)
      : GlslAbstractPropWriter(std::move(parent_writer)) {}

  std::string getTemplateStr() const final { return parent_writer_->getTemplateStr(); }

  std::string getFinalStr() const final {
    return "decompress_geo_coord_" + prop_->getName().substr(0, 1) + "(" +
           parent_writer_->getFinalStr() + ")";
  }
};

class PackedPixelCoordWriter : public GlslAbstractPropWriter {
 public:
  PackedPixelCoordWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer)
      : GlslAbstractPropWriter(std::move(parent_writer)) {}

  std::string getTemplateStr() const final {
    return "get" + prop_->getName() + "(project" + prop_->getName() + "(" +
           parent_writer_->getTemplateStr() + "))";
  }

  std::string getFinalStr() const final {
    return "unpack_pixel_coord_" + prop_->getName().substr(0, 1) + "(" +
           parent_writer_->getFinalStr() + ")";
  }
};

class ProjectWriter : public GlslAbstractPropWriter {
 public:
  ProjectWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer,
                const bool has_proj)
      : GlslAbstractPropWriter(std::move(parent_writer)), has_proj_{has_proj} {}

  std::string getTemplateStr() const final {
    return "project" + prop_->getName() + "(" + parent_writer_->getTemplateStr() + ")";
  }

  std::string getFinalStr() const final {
    return (has_proj_
                ? "project" + prop_->getName() + "(" + parent_writer_->getFinalStr() + ")"
                : parent_writer_->getFinalStr());
  }

 private:
  const bool has_proj_;
};

class CastWriter : public GlslAbstractPropWriter {
 public:
  CastWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer,
             const std::string& cast_str)
      : GlslAbstractPropWriter(std::move(parent_writer)), cast_str_{cast_str} {}

  std::string getTemplateStr() const final {
    // there should be no casting done in the templatized shaders
    return parent_writer_->getTemplateStr();
  }

  std::string getFinalStr() const final {
    return cast_str_ + "(" + parent_writer_->getFinalStr() + ")";
  }

 private:
  const std::string cast_str_;
};

class ScaleFunctionWriter : public GlslAbstractPropWriter {
 public:
  ScaleFunctionWriter(std::unique_ptr<GlslAbstractPropWriter>&& parent_writer,
                      const std::string& orig_func_name,
                      const std::string& alt_func_name = "")
      : GlslAbstractPropWriter(std::move(parent_writer))
      , orig_func_name_{orig_func_name}
      , alt_func_name_{alt_func_name} {}

  std::string getTemplateStr() const final {
    return orig_func_name_ + "(" + parent_writer_->getTemplateStr() + ")";
  }

  std::string getFinalStr() const final {
    return (alt_func_name_.size()
                ? alt_func_name_ + "(" + parent_writer_->getFinalStr() + ")"
                : orig_func_name_ + "(" + parent_writer_->getFinalStr() + ")");
  }

 private:
  const std::string orig_func_name_;
  std::string alt_func_name_;
};

}  // namespace QueryRenderer
