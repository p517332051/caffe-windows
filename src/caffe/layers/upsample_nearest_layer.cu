// --------------------------------------------------------
// fpn upsamplenearest 
// Written by hw, 2018.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/upsample_nearest_layer.hpp"


using std::max;
using std::min;

namespace caffe {
	__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
		int x, y, z, w;
		w = ii % d3;
		ii = ii / d3;
		z = ii % d2;
		ii = ii / d2;
		y = ii % d1;
		ii = ii / d1;
		x = ii;
		w = w / scale_factor;
		z = z / scale_factor;
		d2 /= scale_factor;
		d3 /= scale_factor;
		return (((x*d1 + y)*d2) + z)*d3 + w;
	}

	__device__ int translate_idx_inv(
		int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
		int x, y, z, w;
		w = ii % d3;
		ii = ii / d3;
		z = ii % d2;
		ii = ii / d2;
		y = ii % d1;
		ii = ii / d1;
		x = ii;
		w = w*scale_factor + off_x;
		z = z*scale_factor + off_y;
		d2 *= scale_factor;
		d3 *= scale_factor;
		return (((x*d1 + y)*d2) + z)*d3 + w;
	}
	template <typename Dtype>
	__global__ void upscale(const Dtype *input, Dtype *output, long no_elements,
		int scale_factor, int d1, int d2, int d3) {
		long ii = threadIdx.x + blockDim.x * blockIdx.x;
		ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
		if (ii >= no_elements) return;
		int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
		output[ii] = input[ipidx];
	}
	template <typename Dtype>
	__global__ void downscale(Dtype *gradInput_data, const Dtype *gradOutput_data,
		long no_elements, int scale_factor, int d1, int d2,
		int d3) {
		long ii = threadIdx.x + blockDim.x * blockIdx.x;
		ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
		if (ii >= no_elements) return;
		for (int i = 0; i < scale_factor; i++){
			for (int j = 0; j < scale_factor; j++){
				int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
				gradInput_data[ii] += gradOutput_data[ipidx];
			}
		}
	}
	

  

  template <typename Dtype>
  void UpsampleNearestLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int d1;
	int d2;
	int d3;
	if (top[0]->num_axes()==3)
	{
		 d1 = top[0]->shape(0);
		 d2 = top[0]->shape(1);
		 d3 = top[0]->shape(2);
	}
	else
	{
		 d1 = top[0]->shape(1);
		 d2 = top[0]->shape(2);
		 d3 = top[0]->shape(3);
	}
	long no_elements = top[0]->count();
	long nthreads = 256;
	long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
	long n_yblocks = (long)ceil(
		(float)no_elements / (float)(n_xblocks * nthreads));
	CHECK_LE(n_yblocks,65535);
	dim3 blocks(n_xblocks, n_yblocks);
	dim3 threads(nthreads);
	upscale<Dtype> << <blocks,
		threads >> >(
		bottom_data, top_data, no_elements, scale_, d1, d2, d3);

    CUDA_POST_KERNEL_CHECK;
  }    

  

  template <typename Dtype>
  void UpsampleNearestLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	int d1;
	int d2;
	int d3;
	if (bottom[0]->num_axes()==3)
	{
		 d1 = bottom[0]->shape(0);
		 d2 = bottom[0]->shape(1);
		 d3 = bottom[0]->shape(2);
	}
	else
	{
		 d1 = bottom[0]->shape(1);
		 d2 = bottom[0]->shape(2);
		 d3 = bottom[0]->shape(3);
	}
	long no_elements = bottom[0]->count();
	long nthreads = 256;
	long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
	long n_yblocks = (long)ceil(
		(float)no_elements / (float)(n_xblocks * nthreads));
	CHECK_LE(n_yblocks,65535);
	dim3 blocks(n_xblocks, n_yblocks);
	dim3 threads(nthreads);
	caffe_gpu_set(no_elements, Dtype(0), bottom_diff);
	downscale<Dtype> << <blocks,
		threads >> >(
		bottom_diff, top_diff, no_elements, scale_, d1, d2, d3);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(UpsampleNearestLayer);

}  // namespace caffe
