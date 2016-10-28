#include "CardinalityEstimator.h"

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return {ra_exe_unit.input_descs,
          ra_exe_unit.extra_input_descs,
          ra_exe_unit.input_col_descs,
          ra_exe_unit.simple_quals,
          ra_exe_unit.quals,
          ra_exe_unit.join_type,
          ra_exe_unit.join_dimensions,
          ra_exe_unit.inner_join_quals,
          ra_exe_unit.outer_join_quals,
          {},
          {},
          {},
          std::make_shared<const Analyzer::NDVEstimator>(ra_exe_unit.groupby_exprs),
          SortInfo{{}, SortAlgorithm::Default, 0, 0},
          0};
}
