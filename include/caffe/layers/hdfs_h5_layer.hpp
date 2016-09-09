/*
 * hdfs_h5_layer.hpp
 *
 *  Created on: Aug 4, 2016
 *      Author: lgueguen
 */

#ifndef INCLUDE_CAFFE_LAYERS_HDFS_H5_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_HDFS_H5_LAYER_HPP_


#include "hdf5.h"
#include "hdfs.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {


template <typename Dtype>
class HDFSHDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDFSHDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDFSHDF5DataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HDFSHDF5Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadHDF5FileData(const char* filename);
  virtual hdfsFS openHdfsFs();
  virtual size_t HdfsFileSize(hdfsFS fs, const char * fname);
  virtual char * HdfsFileReadBytes(hdfsFS fs, const char * fname, tSize * nbBytes);
  virtual void HdfsFileReadH5Names(hdfsFS fs, const char * fname);

  virtual void RetryLoadHDF5FileData();


  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;
  std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;

  std::string hdfs_config_file;
  std::string hdfs_master_node;
  int hdfs_master_port;
  int max_number_minutes_retry;
};

}



#endif /* INCLUDE_CAFFE_LAYERS_HDFS_H5_LAYER_HPP_ */
