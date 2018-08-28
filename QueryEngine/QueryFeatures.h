#ifndef QUERYFEATURES_H
#define QUERYFEATURES_H

class GeoExprFeature { /* Implement me */
 public:
  bool isPreflightCountRequired() const { return false; }

 private:
};

class ArrayExprFeature {
 public:
  bool isArrayExprPresent() const { return array_expr_presence_; }
  int getArrayExprCount() const { return array_expr_count_; }
  int getAndBumpArrayExprCount() {
    array_expr_presence_ = true;
    return array_expr_count_++;
  };

  // Note: Not needed if checked_malloc in use
  bool isPreflightCountRequired() const { return array_expr_presence_; }

 private:
  bool array_expr_presence_ = false;
  int array_expr_count_ = 0;
};

class ExecutionRestrictions {
 public:
  bool isCPUOnlyExecutionRequired() const { return cpu_only_required_; }
  void setCPUOnlyExecutionRequired() { cpu_only_required_ = true; }

 private:
  bool cpu_only_required_ = false;
};

template <typename... FEATURE_MARKERS>
class QueryFeatureAggregator : public FEATURE_MARKERS... {
 public:
  bool isPreflightCountRequired() const {
    return internalIsPreflightCountRequired<FEATURE_MARKERS...>();
  }

 private:
  template <typename FEATURE_TYPE>
  bool internalIsPreflightCountRequired() const {
    return this->FEATURE_TYPE::isPreflightCountRequired()();
  }

  template <typename FIRST_FEATURE_TYPE,
            typename SECOND_FEATURE_TYPE,
            typename... REMAINING_FEATURES>
  bool internalIsPreflightCountRequired() const {
    return this->FIRST_FEATURE_TYPE::isPreflightCountRequired()() ||
           internalIsPreflightCountRequired<SECOND_FEATURE_TYPE, REMAINING_FEATURES...>();
  }
};

using QueryFeatureDescriptor =
    QueryFeatureAggregator<ArrayExprFeature, GeoExprFeature, ExecutionRestrictions>;

#endif
