#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "pcl_utils.h"


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PclCloud;
typedef utils::PclUtils<PointT> PclUtils;


void showCloud() {
  PclCloud::Ptr cloud (new PclCloud);
  pcl::io::loadPCDFile<PointT>("../data/sample-scene.pcd", *cloud);
  PclUtils::showCloud(cloud);
}


int main() {
  showCloud();
  return 0;
}

