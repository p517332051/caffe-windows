#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {
	
	template <typename Dtype>
	__global__ void RoIAlignForward(
		const int nthreads,
		const Dtype* bottom_data,
		const Dtype spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio,
		const Dtype* bottom_rois,
		Dtype* top_data) {
		CUDA_KERNEL_LOOP(index,nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			const Dtype* offset_bottom_rois = bottom_rois + n * 5;
			int roi_batch_ind = offset_bottom_rois[0];

			// Do not using rounding; this implementation detail is critical
			Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
			Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
			Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
			Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;
			// T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
			// T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
			// T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

			// Force malformed ROIs to be 1x1
			Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)1.);
			Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1.);
			Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

			const Dtype* offset_bottom_data =
				bottom_data + (roi_batch_ind * channels + c) * height * width;

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0)
				? sampling_ratio
				: ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w =
				(sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			Dtype output_val = 0.;
			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const Dtype y = roi_start_h + ph * bin_size_h +
					static_cast<Dtype>(iy + .5f) * bin_size_h /
					static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++) {
					const Dtype x = roi_start_w + pw * bin_size_w +
						static_cast<Dtype>(ix + .5f) * bin_size_w /
						static_cast<Dtype>(roi_bin_grid_w);

					Dtype val = bilinear_interpolate(
						offset_bottom_data, height, width, y, x, index);
					output_val += val;
				}
			}
			output_val /= count;

			top_data[index] = output_val;
		}
	}


template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* bottom_rois = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int* argmax_data = max_idx_.mutable_gpu_data();
	int count = top[0]->count();
	RoIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS >> >(
		count,
		bottom_data,
		spatial_scale_,
		channels_,
		height_,
		width_,
		pooled_height_,
		pooled_width_,
		sampling_ratio_,
		bottom_rois,
		top_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void RoIAlignBackwardFeature(
	const int nthreads,
	const Dtype* top_diff,
	const int num_rois,
	const Dtype spatial_scale,
	const int channels,
	const int height,
	const int width,
	const int pooled_height,
	const int pooled_width,
	const int sampling_ratio,
	Dtype* bottom_diff,
	const Dtype* bottom_rois) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// (n, c, ph, pw) is an element in the pooled output
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int n = index / pooled_width / pooled_height / channels;

		const Dtype* offset_bottom_rois = bottom_rois + n * 5;
		int roi_batch_ind = offset_bottom_rois[0];

		// Do not using rounding; this implementation detail is critical
		Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
		Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
		Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
		Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;
		// T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
		// T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
		// T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
		// T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

		// Force malformed ROIs to be 1x1
		Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)1.);
		Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1.);
		Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
		Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

		Dtype* offset_bottom_diff =
			bottom_diff + (roi_batch_ind * channels + c) * height * width;

		int top_offset = (n * channels + c) * pooled_height * pooled_width;
		const Dtype* offset_top_diff = top_diff + top_offset;
		const Dtype top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

		// We use roi_bin_grid to sample the grid and mimic integral
		int roi_bin_grid_h = (sampling_ratio > 0)
			? sampling_ratio
			: ceil(roi_height / pooled_height); // e.g., = 2
		int roi_bin_grid_w =
			(sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

		// We do average (integral) pooling inside a bin
		const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

		for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
		{
			const Dtype y = roi_start_h + ph * bin_size_h +
				static_cast<Dtype>(iy + .5f) * bin_size_h /
				static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
			for (int ix = 0; ix < roi_bin_grid_w; ix++) {
				const Dtype x = roi_start_w + pw * bin_size_w +
					static_cast<Dtype>(ix + .5f) * bin_size_w /
					static_cast<Dtype>(roi_bin_grid_w);

				Dtype w1, w2, w3, w4;
				int x_low, x_high, y_low, y_high;

				bilinear_interpolate_gradient(
					height,
					width,
					y,
					x,
					w1,
					w2,
					w3,
					w4,
					x_low,
					x_high,
					y_low,
					y_high,
					index);

				Dtype g1 = top_diff_this_bin * w1 / count;
				Dtype g2 = top_diff_this_bin * w2 / count;
				Dtype g3 = top_diff_this_bin * w3 / count;
				Dtype g4 = top_diff_this_bin * w4 / count;

				if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
					caffe_gpu_atomic_add(
						static_cast<Dtype>(g1), offset_bottom_diff + y_low * width + x_low);
					caffe_gpu_atomic_add(
						static_cast<Dtype>(g2), offset_bottom_diff + y_low * width + x_high);
					caffe_gpu_atomic_add(
						static_cast<Dtype>(g3), offset_bottom_diff + y_high * width + x_low);
					caffe_gpu_atomic_add(
						static_cast<Dtype>(g4), offset_bottom_diff + y_high * width + x_high);
				} // if
			} // ix
		} // iy
	} // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward




template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  const int top_count = top[0]->count();
  const int roi_num = bottom[1]->num();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  

  RoIAlignBackwardFeature<Dtype> << <CAFFE_GET_BLOCKS(count),
	  CAFFE_CUDA_NUM_THREADS >> >(
	  top_count,
	  top_diff,
	  roi_num,
	  spatial_scale_,
	  channels_,
	  height_,
	  width_,
	  pooled_height_,
	  pooled_width_,
	  sampling_ratio_,
	  bottom_diff,
	  bottom_rois);


  // NOLINT_NEXT_LINE(whitespace/operators)
  //ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //    count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
  //    height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}
//
INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
