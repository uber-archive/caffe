/*
 * hdfs_h5_layer.cpp
 *
 *  Created on: Aug 4, 2016
 *      Author: lgueguen
 */
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <sstream>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdfs_h5_layer.hpp"
#include "caffe/util/hdf5.hpp"

#include "hdfs.h"

#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem.hpp>

#include <thread>
#include <chrono>

namespace caffe {

tSize BUFFER_SIZE= 100000;

using namespace boost;
using namespace boost::property_tree;


template <typename Dtype>
hdfsFS HDFSHDF5DataLayer<Dtype>::openHdfsFs(){
	struct hdfsBuilder * hb = hdfsNewBuilder();

	std::string coresite = hdfs_config_file + "/core-site.xml";
	std::string hdfssite = hdfs_config_file + "/hdfs-site.xml";

	std::string hdfs_master = "file://";

	if ( boost::filesystem::exists( coresite.c_str()) &&
			boost::filesystem::exists( hdfssite.c_str())	) {

		DLOG(INFO) << "Loading hdfs config files from directory : " << hdfs_config_file;

	    ptree tree;
	    read_xml(coresite.c_str(), tree);
	    const ptree & properties = tree.get_child("configuration");
	    BOOST_FOREACH(const ptree::value_type & f, properties){
	    	std::string name = f.second.get<std::string>("name");
	    	std::string value = f.second.get<std::string>("value");
	    	//hdfsBuilderConfSetStr(hb, name.c_str(), value.c_str());

	    	if((name == "fs.defaultFS")||(name == "fs.default.name")){
	    		hdfs_master = value;
	    	}
	    }

	    ptree treebis;
	    read_xml(hdfssite.c_str(), treebis);
	    const ptree & propertiess = treebis.get_child("configuration");
	    BOOST_FOREACH(const ptree::value_type & f, propertiess){
	    	std::string name = f.second.get<std::string>("name");
	    	std::string value = f.second.get<std::string>("value");

	    	hdfsBuilderConfSetStr(hb, name.c_str(), value.c_str());
	    }

		hdfsBuilderSetNameNode(hb, hdfs_master.c_str());
		hdfsBuilderSetNameNodePort(hb, hdfs_master_port);

		hdfsBuilderSetForceNewInstance(hb);
		hdfsFS fs = hdfsBuilderConnect(hb);
		return fs;

	}else{
		DLOG(INFO) << "loading default hdfs config file " ;
		DLOG(INFO) << "default config is the standrard file:// filesystem";
		hdfsFS fs = hdfsConnect("default",0);
		return fs;
	}
}
template <typename Dtype>
size_t HDFSHDF5DataLayer<Dtype>::HdfsFileSize(hdfsFS fs, const char * fname){
	size_t filesize = 0;

	if (hdfsExists(fs, fname) == 0 ){
    	hdfsFileInfo * finfo = hdfsGetPathInfo(fs, fname);
    	filesize = finfo->mSize;
    	hdfsFreeFileInfo(finfo, 1);
	}
	return filesize;
}
template <typename Dtype>
char * HDFSHDF5DataLayer<Dtype>::HdfsFileReadBytes(hdfsFS fs, const char * fname, tSize * nbBytes){
	nbBytes[0] = HdfsFileSize(fs,fname);
	char * buffer = NULL;
	if(nbBytes[0]){
		hdfsFile file = hdfsOpenFile(fs, fname, O_RDONLY, BUFFER_SIZE, 0, 0);
		buffer = new char[nbBytes[0]];
		for(tSize i=0; i< nbBytes[0]; i+=BUFFER_SIZE){
			tSize locBytes =  std::min( nbBytes[0]-i , BUFFER_SIZE);
			tSize byteslu = hdfsPread(fs, file, i, &(buffer[i]), locBytes);
			if(locBytes != byteslu){
				LOG(FATAL) << "could not read properly the bytes of " << fname;
			}
		}
		hdfsCloseFile(fs, file);
	}
	return  buffer;
}

template <typename Dtype>
void HDFSHDF5DataLayer<Dtype>::HdfsFileReadH5Names(hdfsFS fs, const char * fname){

	if (hdfsExists(fs, fname) == 0 ){
		int numEntries=0;
		hdfsFileInfo * dirinfo = hdfsListDirectory(fs, fname, &numEntries);
		for(int i=0; i<numEntries; ++i){
			std::string fn(dirinfo[i].mName);
			if( fn.substr(fn.find_last_of(".")) == ".h5" ){
				hdf_filenames_.push_back(fn);
			}
		}
		hdfsFreeFileInfo(dirinfo, numEntries);
	}
}

template <typename Dtype>
HDFSHDF5DataLayer<Dtype>::~HDFSHDF5DataLayer<Dtype>() { }

template <typename Dtype>
void HDFSHDF5DataLayer<Dtype>::RetryLoadHDF5FileData(){
    LOG(INFO) << "Loading HDFS HDF5 failed, waiting " << max_number_minutes_retry << " seconds before retry";
    std::this_thread::sleep_for (std::chrono::seconds(max_number_minutes_retry));
    max_number_minutes_retry*=2;
    ++current_file_;
	if (current_file_ == num_files_)
          current_file_ = 0;
    LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
}

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDFSHDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  LOG(INFO) << "Loading HDFS HDF5 file: " << filename;

  // open connection and hdfs files
  char * buf_ptr;
  tSize nbBytes=0;
  try{
    hdfsFS fs = openHdfsFs();
    buf_ptr = HdfsFileReadBytes(fs,filename, &nbBytes);
    hdfsDisconnect(fs);
  }catch(int e){
    LOG(INFO) << "Problem with HDFS: " << filename;
    RetryLoadHDF5FileData();
  }
  if(nbBytes == 0){
    LOG(INFO) << "Problem with HDFS: " << filename;
    RetryLoadHDF5FileData();
  }

  // put the hdfs bytes into h5 memory file
  hid_t file_id;
  try{
    file_id = H5LTopen_file_image( buf_ptr, nbBytes, H5LT_FILE_IMAGE_DONT_COPY & H5LT_FILE_IMAGE_DONT_RELEASE);
  }catch(int e){
    LOG(INFO) << "Problem with H5: " << filename;
    RetryLoadHDF5FileData();
  }
  if (file_id < 0) {
    LOG(INFO) << "Problem with H5: " << filename;
    RetryLoadHDF5FileData();
  }


  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    try{
        hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
    }catch(int e){
        LOG(INFO) << "Problem with H5: " << filename;
        RetryLoadHDF5FileData();
    }
  }

  herr_t status;
  try{
    status = H5Fclose(file_id);
  delete [] buf_ptr;
  }catch(int e){
    LOG(INFO) << "Problem with H5: " << filename;
    RetryLoadHDF5FileData();
  }
  if(status<0){
      LOG(INFO) << "Problem with H5: " << filename;
      RetryLoadHDF5FileData();
  }
  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdfshdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }

  // reset normal retry at 10 seconds
  max_number_minutes_retry = 10;
}

template <typename Dtype>
void HDFSHDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  LOG(INFO) << "setting up hdfs layer ";

  const string& source = this->layer_param_.hdfshdf5_data_param().source();
  hdfs_config_file = this->layer_param_.hdfshdf5_data_param().hdfs_config_dir();
  hdfs_master_port = this->layer_param_.hdfshdf5_data_param().hdfs_master_port();
  max_number_minutes_retry = this->layer_param_.hdfshdf5_data_param().max_number_minutes_retry();

  hdf_filenames_.clear();

  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
	  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
	  std::string line;
	  while (source_file >> line) {
		  hdf_filenames_.push_back(line);
	  }
  } else {
	  hdfsFS fs = openHdfsFs();
	  HdfsFileReadH5Names(fs, source.c_str());
	  hdfsDisconnect(fs);
	  if(hdf_filenames_.size() == 0){
		  LOG(INFO) << "Failed to open source file: " << source;
		  LOG(INFO) << "source can be either a local text file containing hdfs h5 files" ;
		  LOG(INFO) << "or source can be an hdfs directory of h5 files";
		  LOG(FATAL) << "could not do anything with the source given";
	  }
  }
  source_file.close();



  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hdfshdf5_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdfshdf5_data_param().batch_size();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(hdf_blobs_[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }

  // sleep time before retry
  max_number_minutes_retry = 1;
}

template <typename Dtype>
void HDFSHDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdfshdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        ++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.hdfshdf5_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          }
          DLOG(INFO) << "Looping around to first file.";
        }
        LoadHDF5FileData(
            hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
      current_row_ = 0;
      if (this->layer_param_.hdfshdf5_data_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDFSHDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDFSHDF5DataLayer);
REGISTER_LAYER_CLASS(HDFSHDF5Data);

}  // namespace caffe





