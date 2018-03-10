// --------------------------------------------------------
// fpn upsamplenearest 
// Written by hw, 2018.
// --------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/upsample_nearest_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void UpsampleNearestLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	  UpsampleNearestParameter upsample_nearest_param =
		  this->layer_param_.upsample_nearest_param();
	  scale_ = upsample_nearest_param.scale();

  }

  template <typename Dtype>
  void UpsampleNearestLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	vector<int> out_shape;
	for (int i = 0; i < bottom[0]->num_axes();i++)
	{
		out_shape.push_back(bottom[0]->shape(i));
	}
	out_shape[bottom[0]->num_axes() - 1] *= scale_;
	out_shape[bottom[0]->num_axes() - 2] *= scale_;
    top[0]->Reshape(out_shape);

  }

  template <typename Dtype>
  void UpsampleNearestLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void UpsampleNearestLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(UpsampleNearestLayer);
#endif

  INSTANTIATE_CLASS(UpsampleNearestLayer);
  REGISTER_LAYER_CLASS(UpsampleNearest);

}  // namespace caffe
