# pcl-utils
Utility functions for PCL Library.

Using C++ PCL library introduces a lot of boilerplate code. Most of the operations requires redundant parameters. This utility functions helps to write easy and clean code to use PCL.

## Usage
Basic usage is shown below. There are a lot more utility functions inside pcl-utils.

```c++

#include "pcl_utils.h"

int main() {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	
	// Show point cloud in PCLViewer
	utils::PclUtils<pcl::PointXYZ>::showCloud(cloud);
	
	// If same point type is being used, it is good to have an alias
	typedef utils::PclUtils<pcl::PointXYZ> PclUtils;
	
	// Downsample pointcloud using VoxelGrid
	PclUtils::downsample(cloud, cloud, 0.007f);
	
	return 0;
}
```

