// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void StaticDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  //threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  string dropout_file = this->layer_param_.static_dropout_param().dropout_file();
  std::ifstream fd(dropout_file.c_str(), std::ios_base::in);
  string mask_str, buf;
  getline(fd, mask_str);
  //std::vector<std::string> mask_vec;
  LOG(INFO) << "Static dropout mask: " << dropout_file << std::endl;
  std::stringstream ss(mask_str);
  while (ss >> buf)
    mask_vec_.push_back(std::atoi(buf.c_str()));
  fd.close();
  LOG(INFO) << "Init static dropout mask, size: " << mask_vec_.size() << std::endl;
  
  threshold_ = 0.5;
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  
}

template <typename Dtype>
void StaticDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  //LOG(INFO) << "Reshape" << bottom[0]->num() << bottom[0]->channels() 
  //          << bottom[0]->height() << bottom[0]->width() << std::endl;
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void StaticDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i){
    top_data[i] = bottom_data[i] * mask_vec_[i];
  }
  LOG(INFO) << "Forward_cpu" << std::endl;
  /*
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  */
}

template <typename Dtype>
void StaticDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i){
      bottom_diff[i] = top_diff[i] * mask_vec_[i];
    }
    LOG(INFO) << "Backward_cpu" << std::endl;
    /*
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
    */
  }
}


#ifdef CPU_ONLY
STUB_GPU(StaticDropoutLayer);
#endif

INSTANTIATE_CLASS(StaticDropoutLayer);
REGISTER_LAYER_CLASS(StaticDropout);

}  // namespace caffe
