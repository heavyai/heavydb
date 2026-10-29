#pragma once

#include <memory>
#include <string>

namespace gfx {

class EditPanel {
 public:
  explicit EditPanel(const std::string& name) : name_{name} {}
  virtual ~EditPanel() = default;

  const std::string& getName() const { return name_; };
  virtual bool draw() = 0;

 protected:
  std::string name_;
};

using EditPanelUqPtr = std::unique_ptr<EditPanel>;

}  // namespace gfx
