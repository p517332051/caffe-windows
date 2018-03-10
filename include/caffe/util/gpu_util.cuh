#ifndef CAFFE_UTIL_GPU_UTIL_H_
#define CAFFE_UTIL_GPU_UTIL_H_

namespace caffe {

template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__
float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
double caffe_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}



template <typename Dtype>
__device__ Dtype bilinear_interpolate(
	const Dtype* bottom_data,
	const int height,
	const int width,
	Dtype y,
	Dtype x,
	const int index /* index for debug only*/) {
	// deal with cases that inverse elements are out of feature map boundary
	if (y < -1.0 || y > height || x < -1.0 || x > width) {
		// empty
		return 0;
	}

	if (y <= 0) {
		y = 0;
	}
	if (x <= 0) {
		x = 0;
	}

	int y_low = (int)y;
	int x_low = (int)x;
	int y_high;
	int x_high;

	if (y_low >= height - 1) {
		y_high = y_low = height - 1;
		y = (Dtype)y_low;
	}
	else {
		y_high = y_low + 1;
	}

	if (x_low >= width - 1) {
		x_high = x_low = width - 1;
		x = (Dtype)x_low;
	}
	else {
		x_high = x_low + 1;
	}

	Dtype ly = y - y_low;
	Dtype lx = x - x_low;
	Dtype hy = 1. - ly, hx = 1. - lx;
	// do bilinear interpolation
	Dtype v1 = bottom_data[y_low * width + x_low];
	Dtype v2 = bottom_data[y_low * width + x_high];
	Dtype v3 = bottom_data[y_high * width + x_low];
	Dtype v4 = bottom_data[y_high * width + x_high];
	Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

	Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

	return val;
}


template <typename Dtype>
__device__ void bilinear_interpolate_gradient(
	const int height,
	const int width,
	Dtype y,
	Dtype x,
	Dtype& w1,
	Dtype& w2,
	Dtype& w3,
	Dtype& w4,
	int& x_low,
	int& x_high,
	int& y_low,
	int& y_high,
	const int index /* index for debug only*/) {
	// deal with cases that inverse elements are out of feature map boundary
	if (y < -1.0 || y > height || x < -1.0 || x > width) {
		// empty
		w1 = w2 = w3 = w4 = 0.;
		x_low = x_high = y_low = y_high = -1;
		return;
	}

	if (y <= 0) {
		y = 0;
	}
	if (x <= 0) {
		x = 0;
	}

	y_low = (int)y;
	x_low = (int)x;

	if (y_low >= height - 1) {
		y_high = y_low = height - 1;
		y = (Dtype)y_low;
	}
	else {
		y_high = y_low + 1;
	}

	if (x_low >= width - 1) {
		x_high = x_low = width - 1;
		x = (Dtype)x_low;
	}
	else {
		x_high = x_low + 1;
	}

	Dtype ly = y - y_low;
	Dtype lx = x - x_low;
	Dtype hy = 1. - ly, hx = 1. - lx;

	// reference in forward
	// T v1 = bottom_data[y_low * width + x_low];
	// T v2 = bottom_data[y_low * width + x_high];
	// T v3 = bottom_data[y_high * width + x_low];
	// T v4 = bottom_data[y_high * width + x_high];
	// T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

	w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

	return;
}





}  // namespace caffe


#endif  // CAFFE_UTIL_GPU_UTIL_H_
