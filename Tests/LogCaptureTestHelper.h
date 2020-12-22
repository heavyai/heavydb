#pragma once

#include <sstream>

#include <gtest/gtest.h>

#include <boost/log/core.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

//
// Helper RAII class to temporarily capture log messages and test for a specific substring
//
class LogCapture {
 public:
  LogCapture()
      : sink_{boost::make_shared<sink_t>()}
      , sstream_{boost::make_shared<std::stringstream>()} {
    sink_->locked_backend()->add_stream(sstream_);
    boost::log::core::get()->add_sink(sink_);
  }
  ~LogCapture() { boost::log::core::get()->remove_sink(sink_); }

  // Test if the captured log contains compare_str
  // Resets the stringstream to allow repeated use
  testing::AssertionResult contains(std::string_view compare_str) {
    // Grab the current stream then clear it for the next test
    auto buf = sstream_->str();
    sstream_->str(std::string());

    // Look for the substring and return Success or Failure
    if (buf.find(compare_str) == std::string::npos) {
      return testing::AssertionFailure()
             << "Log must contain: '" << std::string(compare_str) << "'";
    } else {
      return testing::AssertionSuccess();
    }
  }

 private:
  using sink_t =
      boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>;
  boost::shared_ptr<sink_t> sink_;
  boost::shared_ptr<std::stringstream> sstream_;
};
