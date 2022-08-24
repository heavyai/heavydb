#include "EmptyDataSource.h"

namespace costmodel {

EmptyDataSource::EmptyDataSource()
    : DataSource(DataSourceConfig{
          .dataSourceName = "EmptyDataSource",
          .supportedDevices = {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU},
          .supportedTemplates = {AnalyticalTemplate::GroupBy,
                                 AnalyticalTemplate::Join,
                                 AnalyticalTemplate::Reduce,
                                 AnalyticalTemplate::Scan}}) {}

Detail::DeviceMeasurements EmptyDataSource::getMeasurements(
    const std::vector<ExecutorDeviceType>& devices,
    const std::vector<AnalyticalTemplate>& templates) {
  return {};
}

}  // namespace costmodel
