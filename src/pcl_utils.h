/**
 * Common Utility functions for PCL Library
 *
 * Authors: Michael Gnanasekar (mgnanase@andrew.cmu.edu)
 */
#ifndef UTILS_PCL_UTILS_H_
#define UTILS_PCL_UTILS_H_

#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>
#include <stdlib.h>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>

// MACRO utilities
#define SSTR(x) static_cast<std::ostringstream&>((std::ostringstream() << std::dec << x )).str()


namespace utils {

/**
 * Utility functions for PCL library
 *
 * Once the class is templated, the static methods act as common utiltiy
 * functions for that specific type unless it accepts another type.
 */
template <typename T>
class PclUtils
{
 private:
  typedef pcl::PointCloud<T> PclCloud;

 public:
  /**
   * Creates a PCL Visualizer with general settings such as black background and
   * added coordinate system (xyz axes).
   * 
   * @return Created PCL Visualizer
   */
  static pcl::visualization::PCLVisualizer::Ptr pclViewer() {
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer);
    viewer->setBackgroundColor(0,0,0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return viewer;
  }

  /**
   * Creates a PCL Visualizer with the input cloud added in it.
   */
  static pcl::visualization::PCLVisualizer::Ptr pclViewer(typename PclCloud::Ptr& cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer = pclViewer();
    viewer->addPointCloud<T>(cloud);
    return viewer;
  }

  /**
   * Opens the PCL Visualizer and shows up the given point cloud.
   * 
   * @param cloud The input pointcloud pointer
   */
  static void showCloud(typename PclCloud::Ptr& cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer = PclUtils::pclViewer(cloud);
    viewer->spin();
  }

  /**
   * Add pointcloud to the visualizer with colored option.
   * Default color is RED
   *
   * @param viewer PCL Visualizer object that is created either using pcl-utils or PCL.
   * @param cloud Input pointcloud pointer.
   * @param r Red component. To choose random color, pass r as any negative number.
   * @param g Green component.
   * @param b Blue component.
   */
  static void addColoredCloud(pcl::visualization::PCLVisualizer::Ptr& viewer,
                              typename PclCloud::Ptr& cloud,
                              float r = 255.0f, float g = 0.0f, float b = 0.0f) {
    // random
    if (r < 0) {
      r = rand() % 255;
      g = rand() % 255;
      b = rand() % 255;
    }
    std::string id = SSTR(rand());

    viewer->addPointCloud<T>(cloud, 
                             pcl::visualization::PointCloudColorHandlerCustom<T>(cloud, r, g, b), id);
  }


  /**
   * Downsample pointcloud using given leaf size. 
   * 
   * Refer pcl::VoxelGrid for details.
   *
   * @param cloud Input pointcloud pointer.
   * @param outcloud Output pointcloud pointer.
   * @param leaf Leaf size for VoxelGrid downsampling (default is 0.005)
   */
  static void downsample(typename PclCloud::Ptr& cloud, typename PclCloud::Ptr& outcloud,
                         const float leaf = 0.005f) {
    pcl::VoxelGrid<T> grid;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(cloud);
    grid.filter(*outcloud);
  }

  /**
   * Surface normal estimation for the given pointcloud
   *
   * Refer pcl::NormalEstimationOMP for details
   *
   * @param cloud Input pointcloud pointer.
   * @param outcloud Output pointcloud pointer.
   * @param radius Radius search parameter for Normal Estimation.
   */
  template <typename PointNT>
  static void estimateNormal(typename pcl::PointCloud<PointNT>::Ptr& cloud, 
                             typename pcl::PointCloud<PointNT>::Ptr& outcloud,
                             const float radius = 0.01) {
    /*
     * NormalEstimation does not work from PointXYZRGB -> PointNormal. Hence the
     * method accepts only PointNormal
     */
    pcl::NormalEstimationOMP<PointNT, PointNT> ne;
    ne.setRadiusSearch(radius);
    ne.setInputCloud(cloud);
    ne.compute(*outcloud);
  }

  /**
   * Crop an organized pointcloud using the given bounding box.
   *
   * @param cloud Input pointcloud pointer.
   * @param topleft Eigen vector for (x,y) value to represent topleft corner of
   * the bounding box.
   * @param bottomright Eigen vector for bottom-right corner.
   * @return Cropped pointcloud
   */
  static typename PclCloud::Ptr cropPointCloud(typename PclCloud::Ptr& cloud, 
                                      Eigen::Vector2i& topleft, Eigen::Vector2i& bottomright)
  {
    int width = bottomright[0] - topleft[0];
    int height = bottomright[1] - topleft[1];

    typename PclCloud::Ptr out_cloud (new PclCloud);
    out_cloud->resize(width * height);
    out_cloud->width = width;
    out_cloud->height = height;
    out_cloud->is_dense = cloud->is_dense;
    out_cloud->header = cloud->header;
    out_cloud->sensor_orientation_ = cloud->sensor_orientation_;
    out_cloud->sensor_origin_ = cloud->sensor_origin_;

    for (int i=topleft[0], c = 0; i < bottomright[0]; i++, c++)
    {
      for (int j = topleft[1], r = 0; j < bottomright[1]; j++, r++)
      {
        (*out_cloud)(c, r) = (*cloud)(i, j);
      }
    }
    return out_cloud;
  }

  /**
   * Crop an organized pointcloud using the given bounding box.
   *
   * @param cloud Input pointcloud pointer.
   * @param box Eigen vector that contains (x,y,x2,y2) denoting a bounding box.
   * @return Cropped pointcloud
   */
  static typename PclCloud::Ptr cropPointCloud(typename PclCloud::Ptr& cloud,
                             const Eigen::Vector4i& box) {
    Eigen::Vector2i topleft(box[0], box[1]);
    Eigen::Vector2i bottomright(box[2], box[3]);
    return cropPointCloud(cloud, topleft, bottomright);
  }

  /**
   * Pass-Through filter for any axis. Filters the points that meets the given
   * minmax criteria. Filter is applied on the input pointcloud itself.
   *
   * @param cloud Input pointcloud pointer.
   * @param axis A string denoting the axis. Either "x" / "y" / "z".
   * @param min Minimum allowed value for the given axis.
   * @param max Maximum allowed value for the given axis.
   */  
  static void filterOutliers(typename PclCloud::Ptr& cloud, std::string axis, float min, float max) {
    pcl::PassThrough<T> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName(axis);
    pass.setFilterLimits(min, max);
    pass.setKeepOrganized(true);
    pass.filter(*cloud);
  }


  /**
   * Indices points extractor. Extract the points from the pointcloud based on the input indices.
   *
   * @param cloud Input pointcloud pointer.
   * @param indices Vector of indices of the points in the pointcloud.
   * @param negative If true, the indices are excluded and remaining points are
   * retained. Default is false.
   * @param keep_organized If true, the points that does not met the condition
   * will be set to NaN.
   * @return Output cloud after filtering.
   */
  static typename PclCloud::Ptr extractIndices(typename PclCloud::Ptr& cloud, 
                                               std::vector<int>& indices,
                                               bool negative = false,
                                               bool keep_organized = false) {
    pcl::PointIndices::Ptr pcl_indices (new pcl::PointIndices);
    pcl_indices->indices = indices;
    return extractIndices(cloud, pcl_indices, negative, keep_organized);
  }
  
  /**
   * Indices points extractor. Extract the points from the pointcloud based on the input indices.
   *
   * @param cloud Input pointcloud pointer.
   * @param indices PointIndices object that represents the input indices.
   * @param negative If true, the indices are excluded and remaining points are
   * retained. Default is false.
   * @param keep_organized If true, the points that does not met the condition
   * will be set to NaN.
   * @return Output cloud after filtering.
   */
  static typename PclCloud::Ptr extractIndices(typename PclCloud::Ptr& cloud, 
                                               pcl::PointIndices::Ptr& indices,
                                               bool negative = false,
                                               bool keep_organized = false) {
    typename PclCloud::Ptr out_cloud (new PclCloud);
    extractIndices(cloud, indices, out_cloud, negative, keep_organized);
    return out_cloud;
  }

  /**
   * Indices points extractor. Extract the points from the pointcloud based on the input indices.
   *
   * @param cloud Input pointcloud pointer.
   * @param indices Vector of indices of the points in the pointcloud.
   * @param outcloud Output pointcloud pointer.
   * @param negative If true, the indices are excluded and remaining points are
   * retained. Default is false.
   * @param keep_organized If true, the points that does not met the condition
   * will be set to NaN.
   */
  static void extractIndices(typename PclCloud::Ptr& cloud, 
                              pcl::PointIndices::Ptr& indices,
                              typename PclCloud::Ptr& out_cloud,
                              bool negative = false,
                              bool keep_organized = false) {
    pcl::ExtractIndices<T> filter;
    filter.setInputCloud(cloud);
    filter.setIndices(indices);
    filter.setNegative(negative);
    filter.setKeepOrganized(keep_organized);
    filter.filter(*out_cloud);
  }

  /**
   * Fits a plane in the given pointcloud using RANSAC.
   *
   * @param cloud Input pointcloud pointer.
   * @param coeff Output plane coefficients of the fitten plane.
   * @param inliers Output point indices of the inlier points.
   */
  static void segmentPlane(typename PclCloud::Ptr& cloud, 
      pcl::ModelCoefficients::Ptr& coeff, 
      pcl::PointIndices::Ptr& inliers)
  {
    pcl::SACSegmentation<T> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);
    
    if (inliers->indices.size() == 0)
    {
      PCL_ERROR("Could not estimate a planar model for the given point cloud");
      return;
    }
  }

  /**
   * Filter out the pointcloud and keep only table-top points.
   *
   * This utility can be used to filter the objects on top of the table.
   * Input cloud is modified directly.
   *
   * @param cloud Input pointcloud pointer.
   * @param min_table_z Minimum z value to be considered.
   * @param min_table_x Minimum x value to be considered.
   */
  static void filterTableTop(typename PclCloud::Ptr& cloud, float min_table_z, float min_table_x) {
    // Remove outliers (keep inliers)
    // Table is retained to fit a plane in next step (hence min_table_z - 0.1)
    filterOutliers(cloud, "z", min_table_z - 0.1, min_table_z + 1);
    filterOutliers(cloud, "x", min_table_x, min_table_x + 1);

    // Fit a plane for table
    pcl::ModelCoefficients::Ptr coeff (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    segmentPlane(cloud, coeff, inliers);

    bool negative = true, keep_organized = true;
    extractIndices(cloud, inliers, cloud, negative, keep_organized);
  }

  /**
   * Convert XYZ cloud to XYZRGB cloud.
   */
  static void convertToXYZRGB(typename pcl::PointCloud<T>& cloud, Eigen::Vector3i& color, 
      pcl::PointCloud<pcl::PointXYZRGB>& out_cloud)
  {
    pcl::copyPointCloud(cloud, out_cloud);

    for (size_t i = 0; i < out_cloud.size(); i++) {
      pcl::PointXYZRGB &p = out_cloud.points[i];
      if (!isnan(p.x)) {
        p.r = color[0];
        p.g = color[1];
        p.b = color[2];
      }
    }
  }

  /**
   * Cluster pointcloud by Euclidean distance.
   *
   * ClusterTolerance: 0.02 // 2cm
   * Min cluster size: 600
   * Max cluster size: 50000
   *
   * @param cloud Input pointcloud pointer.
   * @param cluster_indices Output vector of PointIndices of all the clusters
   */
  static void cluster(typename PclCloud::Ptr& cloud, std::vector<pcl::PointIndices>& cluster_indices) {
    using std::vector;

    pcl::IndicesPtr indices (new vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    typename pcl::search::KdTree<T>::Ptr tree (new pcl::search::KdTree<T>);
    tree->setInputCloud(cloud, indices);

    pcl::EuclideanClusterExtraction<T> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (600);
    ec.setMaxClusterSize (50000);
    ec.setSearchMethod (tree);
    ec.setIndices(indices);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);
  }

  /**
   * Find the enclosing 2D bouding box of the given cluster
   *
   * @param indices Cluster indices in the pointcloud
   * @param cloud Input pointcloud pointer.
   * @return Eigen vector of (x,y,x2,y2) bouding box.
   */
  static Eigen::Vector4i findBox(const pcl::PointIndices& indices, typename PclCloud::Ptr& cloud)
  {
    int minxi = cloud->width, maxxi = 0, minyi = cloud->height, maxyi = 0;
    for (std::vector<int>::const_iterator it = indices.indices.begin(); 
        it != indices.indices.end(); ++it)
    {
      int xi = *it % cloud->width;
      int yi = (int) (*it / cloud->width);

      minxi = std::min(minxi, xi);
      maxxi = std::max(maxxi, xi);
      minyi = std::min(minyi, yi);
      maxyi = std::max(maxyi, yi);
    }
    Eigen::Vector4i range (minxi, minyi, maxxi, maxyi);
    return range;
  }

  /**
   * Finds the closest Non-NaN point around the given location. Searches for the
   * NonNan point in a circular way by increasing radius on each iteration.
   *
   * @param cloud Input pointcloud pointer.
   * @param cx Center point x-value.
   * @param cy Center point y-value.
   * @return The closest point.
   */
  static T findNearestNonNan(typename PclCloud::Ptr& cloud, int cx, int cy) {
    T pointCenter = (*cloud)(cx, cy);
    // search for non-nan point in a circular with increasing radius
    int r = 1;
    while (isnan(pointCenter.x))
    {
      pointCenter = (*cloud)(cx + r, cy);
      if (!isnan(pointCenter.x)) {
        break;
      }

      pointCenter = (*cloud)(cx - r, cy);
      if (!isnan(pointCenter.x)) {
        break;
      }
      pointCenter = (*cloud)(cx, cy + r);
      if (!isnan(pointCenter.x)) {
        break;
      }
      pointCenter = (*cloud)(cx, cy - r);
      r++;
    }
    return pointCenter;
  }

  /**
   * Factory method to create a Point of templated type.
   *
   * @return Newly created point.
   */
  static T createPoint(float x, float y, float z) {
    T point;
    point.x = x;    
    point.y = y;
    point.z = z;
    return point;
  }

  /**
   * Factory method to create pointcloud from the Eigen matrix
   *
   * @param mat Eigen matrix of size 3xn.
   * @return Newly created pointcloud pointer.
   */
  static typename PclCloud::Ptr createCloud(const Eigen::Ref<const Eigen::MatrixXf>& mat) {
    typename PclCloud::Ptr cloud (new PclCloud);

    for (int col=0; col < mat.cols(); ++col) {
      float x = mat(0, col);
      float y = mat(1, col);
      float z = mat(2, col);
      T point = createPoint(x, y, z);
      cloud->push_back(point);
    }
    return cloud;
  }

  /**
   * Creates a 3D oriented bounding box around the given pointcloud.
   *
   * @param cloud Input pointcloud pointer.
   * @return Eigen matrix of size 3x8. Eight point of the cuboid.
   */
  static Eigen::Matrix<float, 3, 8> createOrientedBox(typename PclCloud::Ptr& cloud) {
    pcl::PCA<T> pca;
    pca.setInputCloud(cloud);
    typename PclCloud::Ptr proj_c (new PclCloud);

    pca.project(*cloud, *proj_c);

    T proj_min, proj_max, min, max;
    pcl::getMinMax3D(*proj_c, proj_min, proj_max);

    Eigen::Matrix<float, 3, 8> mat, res;
    mat.col(0) << proj_min.x, proj_min.y, proj_min.z,
    mat.col(1) << proj_max.x, proj_min.y, proj_min.z,
    mat.col(2) << proj_max.x, proj_max.y, proj_min.z,
    mat.col(3) << proj_min.x, proj_max.y, proj_min.z,
    mat.col(4) << proj_min.x, proj_max.y, proj_max.z,
    mat.col(5) << proj_min.x, proj_min.y, proj_max.z,
    mat.col(6) << proj_max.x, proj_min.y, proj_max.z,
    mat.col(7) << proj_max.x, proj_max.y, proj_max.z;
            
    for (int i=0; i < 8; ++i) {
      T proj_point = createPoint(mat(0, i), mat(1, i), mat(2, i));
      T point;
      pca.reconstruct(proj_point, point);
      res(0, i) = point.x;
      res(1, i) = point.y;
      res(2, i) = point.z;
    }

    return res;
  }

};  // class PclUtils

}  // namespace utils

#endif  // UTILS_PCL_UTILS_H_
