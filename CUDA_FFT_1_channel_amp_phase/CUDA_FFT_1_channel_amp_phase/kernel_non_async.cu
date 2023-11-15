#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include<time.h>

#define NUM_STREAMS 4
#define NUM_BLINES 40
#define NBLINES 960
// Complex data type
typedef float2 Complex;

//static __device__ __host__ inline void ComplexPointwiseCalc(float *signal, float *temp_signal, int k);
static __global__ void ComplexPointwiseCalc(float *d_trimmed_signal, float *d_FFT_signal, int num_aline, int start, int depth, uint16_t *curve);
static __global__ void PadData(float *, uint16_t *, int num_aline);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(uint16_t *h_signal, float *h_trimmed_signal, uint16_t *d_signal, float *d_padded_signal, float *d_FFT_signal, float *d_trimmed_signal, cufftHandle *plans, cudaStream_t *streams, int num_aline, int start, int depth, uint16_t *curve);

extern "C" __declspec(dllexport)void multi_runtest(uint16_t *h_signal, float *h_trimmed_signal, int num_aline, int start, int depth, uint16_t *curve);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	uint16_t *h_signal;
	clock_t start, stop;
	h_signal = (uint16_t *)malloc(sizeof(uint16_t) * 1440 * 2200 * NBLINES);
	float *h_trimmed_signal;
	h_trimmed_signal = (float *)malloc(sizeof(float) * 200 * 2200 * NBLINES);

	//for (int i = 0; i < 1440 * 2000* 700; i++) {
	//	h_signal[i] = rand();
	//}
	//printf("%f\n", (h_signal + 19 * 1440 * 2000 * 40)[0]);
	uint16_t *curve;
	curve = (uint16_t *)malloc(sizeof(uint16_t) * 1100 * NBLINES);
	memset(curve, 1, 1100 * NBLINES * sizeof(uint16_t));

	printf("start kernel function\n");
	start = clock();
	multi_runtest(h_signal, h_trimmed_signal, 1100, 50, 100, curve);

	stop = clock();
	printf("time for whole program is %f s\n", (double)(stop - start) / CLOCKS_PER_SEC);
	printf("%f\n", h_trimmed_signal[10]);
	printf("%f\n", h_trimmed_signal[12]);
	printf("%f\n", h_trimmed_signal[14]);
	printf("%f\n", h_trimmed_signal[16]);
	printf("%f\n", h_trimmed_signal[18]);
	printf("%f\n", h_trimmed_signal[20]);
	printf("%f\n", h_trimmed_signal[22]);
	printf("%f\n", h_trimmed_signal[24]);
	printf("%f\n", h_trimmed_signal[26]);
	//printf("h_padded_signal %f \n", h_padded_signal[1]);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void multi_runtest(uint16_t *h_signal, float *h_trimmed_signal, int num_aline, int start, int depth, uint16_t *curve) {
	//create streams
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) { checkCudaErrors(cudaStreamCreate(&streams[i])); }
	//create plan for FFT and assign each plan to different stream
	cufftHandle* plans = (cufftHandle*)malloc(sizeof(cufftHandle)*NUM_STREAMS);
	for (int i = 0; i < NUM_STREAMS; i++) {
		cufftPlan1d(&plans[i], 2048, CUFFT_R2C, 2 * num_aline * NUM_BLINES / NUM_STREAMS);
		//cufftSetStream(plans[i], streams[i]);
	}
	//init device memory
	uint16_t *d_signal;
	float *d_padded_signal;
	float *d_trimmed_signal;
	float *d_FFT_signal;
	uint16_t *d_curve;
	checkCudaErrors(cudaMalloc((void **)&d_signal, sizeof(uint16_t) * 1440 * 2 * num_aline * NUM_BLINES));
	checkCudaErrors(cudaMalloc((void **)&d_padded_signal, sizeof(float) * 2048 * 2 * num_aline * NUM_BLINES));
	checkCudaErrors(cudaMalloc((void **)&d_FFT_signal, sizeof(float) * 2050 * 2 * num_aline * NUM_BLINES));
	checkCudaErrors(cudaMalloc((void **)&d_trimmed_signal, sizeof(float) * 2 * depth * 2 * num_aline * NUM_BLINES));
	checkCudaErrors(cudaMalloc((void **)&d_curve, sizeof(uint16_t) *NBLINES*num_aline));

	checkCudaErrors(cudaMemcpy(d_curve, curve, sizeof(uint16_t) *NBLINES * num_aline, cudaMemcpyHostToDevice));

	int block_size = 320;
	//divide dataset to 5 small groups of 100 blines
	for (int k = 0; k < NBLINES/ block_size; k++) {
		//intf("k is %d\n", k);
		//for async memcpy, memory needs to be registerred first
		//checkCudaErrors(cudaHostRegister(h_signal + k * 1440 * 2 * num_aline * block_size, sizeof(uint16_t) * 1440 * 2 * num_aline * block_size, cudaHostRegisterPortable));
		//checkCudaErrors(cudaHostRegister(h_trimmed_signal + k * 2 * depth * 2 * num_aline * block_size, sizeof(float) * 2 * depth * 2 * num_aline * block_size, cudaHostRegisterPortable));
		//divide each group to 2 sub group with 50 blines
		for (int j = 0; j < block_size/NUM_BLINES; j++) {
			//ntf("j is %d\n", j);
			checkCudaErrors(cudaMemset(d_padded_signal, 0, NUM_BLINES * sizeof(float) * 2048 * 2 * num_aline));
			runTest(h_signal + k * 1440 * 2 * num_aline * block_size + j * 1440 * 2 * num_aline * NUM_BLINES, h_trimmed_signal + k * 2 * depth * 2 * num_aline * block_size + j * 2 * depth * 2 * num_aline * NUM_BLINES, d_signal, d_padded_signal, d_FFT_signal, d_trimmed_signal, plans, streams, num_aline, start, depth, d_curve + k * num_aline * block_size + j * num_aline * NUM_BLINES);
		}
		//checkCudaErrors(cudaHostUnregister(h_signal + k * 1440 * 2 * num_aline * block_size));
		//checkCudaErrors(cudaHostUnregister(h_trimmed_signal + k * 2 * depth * 2 * num_aline * block_size));
	}

	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(d_trimmed_signal));
	checkCudaErrors(cudaFree(d_padded_signal));
	for (int i = 0; i < NUM_STREAMS; i++) {
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}
	cudaDeviceReset();
}

void runTest(uint16_t *h_signal, float *h_trimmed_signal, uint16_t *d_signal, float *d_padded_signal, float *d_FFT_signal, float *d_trimmed_signal, cufftHandle *plans, cudaStream_t *streams, int num_aline, int start, int depth, uint16_t *curve) {
	//findCudaDevice(argc, (const char **)argv);

	//init streams
	for (int i = 0; i < NUM_STREAMS; i++) {
		//printf("i is %d      ", i);
		checkCudaErrors(cudaMemcpy(d_signal + i * 1440 * 2 * num_aline * NUM_BLINES / NUM_STREAMS, h_signal + i * 1440 * 2 * num_aline * NUM_BLINES / NUM_STREAMS, 1440 * 2 * num_aline * sizeof(uint16_t) *NUM_BLINES / NUM_STREAMS, cudaMemcpyHostToDevice)); //use same or different streams for memcopy and computation
		//Pad Data
		//checkCudaErrors(cudaEventRecord(Start, 0));
		//dim3 threadsPerBlock(32, 32);

		PadData << <16, 1024, 0>> > (d_padded_signal + i * 2048 * 2 * num_aline * NUM_BLINES / NUM_STREAMS, d_signal + i * 1440 * 2 * num_aline * NUM_BLINES / NUM_STREAMS, num_aline);
		//checkCudaErrors(cudaEventRecord(Stop, 0));
		//cudaEventSynchronize(Stop);
		//checkCudaErrors(cudaEventElapsedTime(&ms, Start, Stop));
		//printf("time for zero padding : %f ms\n", ms);

		// Transform signal and kernel
		//checkCudaErrors(cudaEventRecord(Start, 0));
		checkCudaErrors(cufftExecR2C(plans[0], reinterpret_cast<cufftReal *>(d_padded_signal + i * 2048 * 2 * num_aline * NUM_BLINES / NUM_STREAMS),
			reinterpret_cast<cufftComplex *>(d_FFT_signal + i * 2050 * 2 * num_aline * NUM_BLINES / NUM_STREAMS)));

		//checkCudaErrors(cudaEventRecord(Stop, 0));
		//cudaEventSynchronize(Stop);
		//checkCudaErrors(cudaEventElapsedTime(&ms, Start, Stop));
		//printf("time for FFT with cuda event timer: %f ms\n", ms);


		// Multiply the coefficients together and normalize the result
		//checkCudaErrors(cudaEventRecord(Start, 0));
		ComplexPointwiseCalc << <16, 1024, 0 >> > (d_trimmed_signal + i * 2 * depth * 2 * num_aline * NUM_BLINES / NUM_STREAMS, d_FFT_signal + i * 2050 * 2 * num_aline * NUM_BLINES / NUM_STREAMS, num_aline, start, depth, curve + i * num_aline * NUM_BLINES / NUM_STREAMS);
		//checkCudaErrors(cudaEventRecord(Stop, 0));
		//cudaEventSynchronize(Stop);
		//cudaEventElapsedTime(&ms, Start, Stop);
		//printf("time for pointwise calculation: %f ms\n", ms);

		// Check if kernel execution generated and error
		//getLastCudaError("Kernel execution failed [ ComplexPointwiseCalc ]");
		checkCudaErrors(cudaMemcpy(h_trimmed_signal + i * 2 * depth * 2 * num_aline * NUM_BLINES / NUM_STREAMS, d_trimmed_signal + i * 2 * depth * 2 * num_aline * NUM_BLINES / NUM_STREAMS, depth * 2 * 2 * num_aline * sizeof(float) * NUM_BLINES / NUM_STREAMS, cudaMemcpyDeviceToHost));
	}
	//printf("\n");

	for (int i = 0; i < NUM_STREAMS; i++) {
		//printf("i is %d      ", i);
		checkCudaErrors(cudaStreamSynchronize(streams[i]));
	}

}

// Pad data



////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

static __global__ void ComplexPointwiseCalc(float *d_trimmed_signal, float *d_FFT_signal, int num_aline, int start, int depth, uint16_t *curve) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int d = threadID; d < depth * 2 * num_aline * NUM_BLINES / NUM_STREAMS; d += numThreads) {
		int pos = d / depth * 1025 + d % depth + start*2 +curve[d / depth / 2];
		float cx = pow(d_FFT_signal[2 * pos], 2) + pow(d_FFT_signal[2 * pos + 1], 2);
		float cy = atan(d_FFT_signal[2 * pos + 1] / d_FFT_signal[2 * pos]);
		d_trimmed_signal[2 * d] = cx;
		d_trimmed_signal[2 * d + 1] = cy;
	}
}

static __global__ void PadData(float *new_data, uint16_t *old_data, int num_aline) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	//const int numThreads = blockDim.x * blockDim.y * gridDim.x;
	//const int threadID = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;

	for (int i = threadID; i < 2880 * num_aline * NUM_BLINES / NUM_STREAMS; i += numThreads) {
		new_data[i % 1440 + i / 1440 * 2048] = old_data[i];

	}

}