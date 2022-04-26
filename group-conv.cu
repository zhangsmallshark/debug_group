#include <cudnn.h>
#include <stdio.h>
#include <cuda.h>
#include <malloc.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

#define B 128
#define R 3
#define S 3
#define TW 2
#define INTERNAL_TH 2
#define INTERNAL_TW 4
#define TH 2
#define TC 1 
#define H 28
#define W 28
#define C 128
#define N 128
#define TB 1

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<endl;
        exit(-1);
    }
}

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

class Conv{
public:
    unsigned int PAD;
    unsigned int hOut;
    unsigned int wOut;
    float *cpuKernel;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    float *output;
    float *kernel;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride);
    float *forward(float *input);
};

void Conv::initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                      unsigned int pad,unsigned int r,unsigned int s,unsigned int stride){

    this->hOut = (H+2*pad - r)/stride + 1;
    this->wOut = (W+2*pad - s)/stride + 1;
    cudaMalloc(&kernel,sizeof(float)*C*N*r*s);
    cudaMalloc(&this->output,sizeof(float)*b*hOut*wOut*N);
    cudnnCreate(&convCudnn);
    cudnnCreateTensorDescriptor(&convInputDescriptor);
    cudnnSetTensor4dDescriptor(convInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/b,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnCreateFilterDescriptor(&convKernelDescriptor);
    cudnnSetFilter4dDescriptor(convKernelDescriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/N,
            /*in_channels=*/C,
            /*kernel_height=*/r,
            /*kernel_width=*/s);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc,
            /*pad_height=*/pad,
            /*pad_width=*/pad,
            /*vertical_stride=*/stride,
            /*horizontal_stride=*/stride,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    int batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          convInputDescriptor,
                                          convKernelDescriptor,
                                          &batch_size,
                                          &channels,
                                          &height,
                                          &width);
    cudnnCreateTensorDescriptor(&convOutputDescriptor);
    cudnnSetTensor4dDescriptor(convOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/N,
            /*image_height=*/hOut,
            /*image_width=*/wOut);
    cudnnGetConvolutionForwardWorkspaceSize(convCudnn,
                                            convInputDescriptor,
                                            convKernelDescriptor,
                                            convDesc,
                                            convOutputDescriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                            &workspace_bytes);
    cudaMalloc(&d_workspace, workspace_bytes);
    unsigned int kernelSize = r*s*C*N;//kernel
    this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
    for(int i=0;i<kernelSize;++i){
        this->cpuKernel[i] = 1.0f;
    }
    cudaMemcpy(kernel,cpuKernel,r*s*C*N*sizeof(float),cudaMemcpyHostToDevice);
    free(cpuKernel);
}

float * Conv::forward(float *input) {
    cudaMemset(output, 0, B*N*hOut*wOut*sizeof(float));
    checkCUDNN(cudnnConvolutionForward(convCudnn,
                                       &alpha,
                                       convInputDescriptor,
                                       input,
                                       convKernelDescriptor,
                                       kernel,
                                       convDesc,
                                       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       convOutputDescriptor,
                                       output));
    return output;
}

__device__ void switch_function(int switch_condition,float *temp_kernel,float v,float *temp_result){
	switch (switch_condition) {
		case 0:
			for ( int r = 0; r < 1; r++) {
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*4+(0-s)] += result;
				}
			}
		break;
		case 1:
			for ( int r = 0; r < 1; r++) {
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*4+(1-s)] += result;
				}
			}
		break;
		case 2:
			for ( int r = 0; r < 1; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*4+(2-s)] += result;
				}
			}
		break;
		case 3:
			for ( int r = 0; r < 1; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*4+(3-s)] += result;
				}
			}
		break;
		case 4:
			for ( int r = 0; r < 1; r++) {
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*4+(4-s)] += result;
				}
			}
		break;
		case 5:
			for ( int r = 0; r < 1; r++) {
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*4+(5-s)] += result;
				}
			}
		break;
		case 6:
			for ( int r = 0; r < 2; r++) {
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*4+(0-s)] += result;
				}
			}
		break;
		case 7:
			for ( int r = 0; r < 2; r++) {
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*4+(1-s)] += result;
				}
			}
		break;
		case 8:
			for ( int r = 0; r < 2; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*4+(2-s)] += result;
				}
			}
		break;
		case 9:
			for ( int r = 0; r < 2; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*4+(3-s)] += result;
				}
			}
		break;
		case 10:
			for ( int r = 0; r < 2; r++) {
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*4+(4-s)] += result;
				}
			}
		break;
		case 11:
			for ( int r = 0; r < 2; r++) {
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*4+(5-s)] += result;
				}
			}
		break;
		case 12:
			for ( int r = 1; r < 3; r++) {
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*4+(0-s)] += result;
				}
			}
		break;
		case 13:
			for ( int r = 1; r < 3; r++) {
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*4+(1-s)] += result;
				}
			}
		break;
		case 14:
			for ( int r = 1; r < 3; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*4+(2-s)] += result;
				}
			}
		break;
		case 15:
			for ( int r = 1; r < 3; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*4+(3-s)] += result;
				}
			}
		break;
		case 16:
			for ( int r = 1; r < 3; r++) {
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*4+(4-s)] += result;
				}
			}
		break;
		case 17:
			for ( int r = 1; r < 3; r++) {
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*4+(5-s)] += result;
				}
			}
		break;
		case 18:
			for ( int r = 2; r < 3; r++) {
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*4+(0-s)] += result;
				}
			}
		break;
		case 19:
			for ( int r = 2; r < 3; r++) {
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*4+(1-s)] += result;
				}
			}
		break;
		case 20:
			for ( int r = 2; r < 3; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*4+(2-s)] += result;
				}
			}
		break;
		case 21:
			for ( int r = 2; r < 3; r++) {
				for ( int s = 0; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*4+(3-s)] += result;
				}
			}
		break;
		case 22:
			for ( int r = 2; r < 3; r++) {
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*4+(4-s)] += result;
				}
			}
		break;
		case 23:
			for ( int r = 2; r < 3; r++) {
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*4+(5-s)] += result;
				}
			}
		break;

	}
}

__device__ void load_input(float * input, float * shared_input, int h_start, int h_end,
                           int c_id, int h_offset, int batch_start, int *channels){
    unsigned int h_len = h_end - h_start;
    for(int i=threadIdx.x;i<h_len*W*TB;i+=blockDim.x){
        int batch_id = batch_start + i/(h_len*W);
        int c = channels[c_id];
        int hw_id = i % (h_len*W);
        int h = hw_id/W;
        int w = hw_id%W;
        int local_h = h+h_offset;
        int local_w = w+1;
        int local_h_no_padding = h_start+h;
        int local_w_no_padding = w;
        shared_input[(batch_id - batch_start)*(TH+2)*(W+2) + local_h*(W+2)+local_w]
                = input[c*(H)*(W)+local_h_no_padding*(W)+local_w_no_padding];
    }
}
__global__ void conv2d_no_padding(float * __restrict__ input,const float * __restrict__ kernel,
                                  float * __restrict__ outputs, int *channels, int *filters, int *channel_ptr,
                                  int *filter_ptr){
    __shared__ float input_tile[(TH+2)*(W+2)*TB];
    __shared__ float shared_kernel[9*N];
    int THS = (H - 1)/TH + 1;
    int TBS = (B - 1)/TB + 1;
    int group_id = blockIdx.x / (TBS*THS);
    int num_channels = channel_ptr[group_id+1] - channel_ptr[group_id];
    int num_filters = filter_ptr[group_id+1] - filter_ptr[group_id];

    int blk_id = blockIdx.x % (TBS*THS);
    int batch_start = (blk_id / THS)*TB;
    int h_start =  (blk_id % THS) * TH;
    float local_compute[INTERNAL_TH*INTERNAL_TW] = {0.0f};
    int block_th = min(TH,(H - h_start));
    int inner_ths = (block_th - 1)/INTERNAL_TH + 1;
    int inner_tws = (W - 1)/INTERNAL_TW + 1;
    int h_copy_start = max(h_start - 1, 0);
    int h_copy_end = min(h_start + block_th + 1, H);
    int h_offset = ((h_start - 1) < 0)?1:0;
    for(unsigned int i=threadIdx.x;i<(TH+2)*(W+2)*TB;i+=blockDim.x){
        input_tile[i] = 0.0f;
    }
    for(int i = threadIdx.x; i<9*N; i+=blockDim.x){
        shared_kernel[i] = 1.0f;
    }
    __syncthreads();
    for(int c=0;c<num_channels;c++){
        int channel_index = channels[channel_ptr[group_id] + c];
        load_input(input,input_tile,h_copy_start,h_copy_end,channel_index,h_offset,batch_start,channels);
        __syncthreads();
        for(int i=threadIdx.x;i<inner_ths*inner_tws*TB;i+=blockDim.x) {
            int b = i / (inner_ths * inner_tws);
            int batch_id = b + batch_start;
            int local_h = ((i % (inner_ths * inner_tws)) / inner_tws) * INTERNAL_TH;
            int local_w = ((i % (inner_ths * inner_tws)) % inner_tws) * INTERNAL_TW;
            int h_end = min(block_th - local_h + 2, INTERNAL_TH + 2);
            int w_end = min(W - local_w + 2, INTERNAL_TW + 2);
            for(int n=0;n<num_filters;++n){
                int output_channel_index = filters[filter_ptr[group_id]+n];
                float *local_kernel = &shared_kernel[0];
                for(int h=0;h<h_end;++h){
                    for(int w=0;w<w_end;++w){
                        float v = input_tile[b*(TH+2)*(W+2)+(local_h+h)*(W+2)+local_w+w];
                        int linear_id = h*(INTERNAL_TW+2) + w;
                        switch_function(linear_id,local_kernel,v,local_compute);
                    }
                }
                for(int h=0;h<INTERNAL_TH;h++){
                    for(int w=0;w<INTERNAL_TW;++w){
                        int h_out = h_start + h + local_h;
                        int w_out = w + local_w;
                        if(h_out>=h_start+block_th||w_out>=W){
                            continue;
                        }
                        atomicAdd(&outputs[batch_id*N*H*W+output_channel_index*H*W+h_out*W+w_out],
                                  local_compute[h*INTERNAL_TW+w]);
                    }
                }
                for(int j=0;j<INTERNAL_TW*INTERNAL_TH;++j){
                    local_compute[j] = 0.0f;
                }
            }
        }
    }
}

float check_diff(float *x, float *y, unsigned int size){
    float diff = 0.0f;
    for(unsigned int i=0;i<size;++i){
        diff += abs(x[i] - y[i]);
    }
    return diff;
}

#define PTR_S 5
#define C_S 328
#define F_S 128

int main(int argc, char *argv[]){
    int groups = 4;
    int filters_ptr[5] = {0, 32, 64, 96, 128};
    int filters[128] = {0, 5, 9, 13, 14, 17, 20, 21, 23, 25, 26, 30, 34, 36, 39, 51, 57, 65, 70, 84, 89, 96, 97, 98, 99, 101, 109, 110, 118, 119, 120, 124, 1, 2, 8, 12, 15, 27, 31, 34, 35, 41, 46, 50, 52, 53, 54, 56, 69, 73, 80, 81, 82, 90, 92, 95, 104, 105, 107, 108, 114, 115, 123, 126, 3, 6, 7, 22, 28, 29, 32, 38, 44, 55, 58, 64, 66, 67, 68, 74, 75, 76, 79, 83, 85, 86, 88, 91, 94, 102, 103, 106, 113, 116, 122, 125, 4, 10, 11, 16, 18, 19, 24, 33, 40, 42, 43, 45, 47, 48, 49, 59, 60, 61, 62, 63, 71, 72, 77, 78, 87, 93, 100, 111, 112, 117, 121, 127};
    int channels_ptr[5] = {0, 71, 146, 239, 328};
    int channels[328] = {0, 1, 2, 5, 6, 10, 11, 12, 14, 17, 20, 22, 23, 24, 25, 26, 30, 33, 35, 36, 37, 38, 39, 40, 41, 43, 44, 48, 52, 57, 58, 59, 61, 62, 63, 64, 65, 68, 69, 71, 72, 74, 75, 76, 78, 79, 80, 85, 86, 87, 88, 90, 91, 92, 95, 96, 98, 99, 100, 101, 103, 107, 110, 111, 113, 114, 115, 117, 119, 120, 121, 1, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 22, 27, 28, 29, 31, 33, 35, 36, 37, 40, 42, 43, 46, 48, 49, 51, 52, 53, 55, 58, 59, 61, 64, 65, 66, 67, 68, 72, 73, 74, 76, 77, 78, 79, 81, 82, 83, 88, 89, 90, 94, 97, 98, 99, 101, 102, 103, 104, 105, 107, 108, 110, 111, 112, 113, 117, 118, 119, 120, 123, 124, 125, 126, 127, 3, 5, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 28, 29, 30, 31, 33, 35, 38, 39, 41, 42, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 58, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 73, 74, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 115, 116, 118, 119, 122, 123, 124, 125, 126, 127, 0, 1, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 41, 42, 45, 46, 49, 50, 51, 52, 54, 60, 61, 62, 64, 65, 66, 67, 70, 71, 75, 76, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 91, 93, 94, 95, 97, 98, 99, 100, 101, 102, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 117, 120, 121, 122, 125, 126, 127};

    int *device_channels;
    int *device_filters;
    int *device_channels_ptr;
    int *device_filters_ptr;

    cudaMalloc(&device_filters,F_S*sizeof(int));
    cudaMalloc(&device_channels,C_S*sizeof(int));
    cudaMalloc(&device_channels_ptr,PTR_S*sizeof(int));
    cudaMalloc(&device_filters_ptr,PTR_S*sizeof(int));

    cudaMemcpy(device_filters,filters,F_S*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_channels,channels,C_S*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_channels_ptr,channels_ptr,PTR_S*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_filters_ptr,filters_ptr,PTR_S*sizeof(int),cudaMemcpyHostToDevice);
    
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    float *input = new float[B*C*H*W];
    time_t t;
    srand((unsigned) time(&t));
    for(int i =0;i<B*C*H*W;++i){
        input[i] = 1.0f;
    }
    float *device_input;
    cudaMalloc(&device_input,B*C*H*W*sizeof(float));
    cudaMemcpy(device_input,input,B*C*H*W*sizeof(float),cudaMemcpyHostToDevice);

    float *K = new float[C*N*9];
    for(int i=0;i<C*N*9;++i){
        K[i] = 0.0f;
    }
    float *device_k;
    cudaMalloc(&device_k,C*N*9*sizeof(float));
    float *out;
    cudaMalloc(&out,B*N*H*W*sizeof(float));

    for(int i=0;i<groups;++i){
        for(int j = filters_ptr[i];j<filters_ptr[i+1];j++){
            int filter = filters[j];
            for(int k = channels_ptr[i];k<channels_ptr[i+1];k++){
                int channel = channels[k];
                for(int r=0;r<3;++r){
                    for(int s=0;s<3;++s){
                        K[filter*C*9+channel*9+r*3+s] = 1.0f;
                    }
                }
            }
        }
    }
    Conv conv;
    conv.initialize(B,C,H,W,N,1,3,3,1);
    cudaMemcpy(conv.kernel,K,C*N*9*sizeof(float),cudaMemcpyHostToDevice);
    float *out_cudnn = conv.forward(device_input);
    float *out_cudnn_host = new float[N*H*W*B];
    cudaMemcpy(out_cudnn_host,out_cudnn,B*N*H*W*sizeof(float),cudaMemcpyDeviceToHost);

    unsigned int gridDim = ((H - 1)/TH + 1)*((B - 1)/TB + 1);
    unsigned int bdim = ((TH - 1)/INTERNAL_TH + 1)*((W - 1)/INTERNAL_TW + 1)*TB;
    cudaEventRecord(event_start);
    cudaMemset(out,0,N*H*W*sizeof(float));
    conv2d_no_padding<<<gridDim*groups,bdim>>>(device_input,device_k,out,device_channels,
                                               device_filters,device_channels_ptr,device_filters_ptr);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_tdc;
    cudaEventElapsedTime(&time_tdc, event_start, event_stop);
    chkerr(cudaDeviceSynchronize());
    chkerr(cudaGetLastError());
    float *out_tdc = new float[B*N*H*W];
    cudaMemcpy(out_tdc,out,B*N*H*W*sizeof(float),cudaMemcpyDeviceToHost);

    cudaEventRecord(event_start);
    out_cudnn = conv.forward(device_input);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_cudnn;
    cudaEventElapsedTime(&time_cudnn, event_start, event_stop);
    cout<<C<<","<<N<<","<<H<<","<<W<<","<<TH<<","<<INTERNAL_TH<<","<<INTERNAL_TW<<","<<
        time_cudnn<<","<<time_tdc<<","<< check_diff(out_cudnn_host,out_tdc,B*N*H*W)<<endl;

    string out_file = "res0.txt";
    ofstream out_s;
    if (time_tdc < time_cudnn) {
        out_s.open(out_file, ios::binary | ios::app | ios::in | ios::out);
        out_s<<C<<","<<N<<","<<H<<","<<W<<","<<TH<<","<<INTERNAL_TH<<","<<INTERNAL_TW<<","<<time_cudnn<<","<<time_tdc<<","<< check_diff(out_cudnn_host,out_tdc,B*N*H*W)<<"\n";
        out_s.close();
        cout << "Find it " << endl;
    }

    return 0;
}