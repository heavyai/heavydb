#include "CardinalityEstimator.h"

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  auto ndv_ra_exe_unit = ra_exe_unit;
  ndv_ra_exe_unit.groupby_exprs = {};
  ndv_ra_exe_unit.target_exprs = {};
  ndv_ra_exe_unit.orig_target_exprs = {};
  ndv_ra_exe_unit.estimator = std::make_shared<const Analyzer::NDVEstimator>(ra_exe_unit.groupby_exprs);
  return ndv_ra_exe_unit;
}
