#ifndef CAFFE_SPATIAL_TRANSFORM_LAYER_HPP_
#define CAFFE_SPATIAL_TRANSFORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SpatialTransformLayer : public Layer<Dtype> {
 public:
  explicit SpatialTransformLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialTransform"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 private:
  inline Dtype max(Dtype a, Dtype b){
    if(a >= b){
	return a;
    }else{
        return b;
    }
  }

  inline Dtype abs(Dtype a){
      if(a >= 0){
	return a;
      }else{
	return -a;
      }
  }

  shared_ptr<Blob<Dtype> > output_grid;
  shared_ptr<Blob<Dtype> > input_grid;

  int num;
  int channel;
  int height;
  int width;

  int output_height;
  int output_width;

  int point_cnt;//bottom[0]->width()*bottom[0]->height()
  int x_cnt;//bottom[0]->height()
  int y_cnt;//bottom[0]->width()

};

}  // namespace caffe

#endif  // CAFFE_SPATIAL_TRANSFORM_LAYER_HPP_
