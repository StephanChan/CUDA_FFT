//process single channel of PSOCT and outputs amplitdue

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <math.h>
//#include "cuPrintf.cu"
//#include "cuPrintf.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#define SWAP(a,b) tempr=(a); (a)=(b); (b)=tempr
#define data_t float
#define data_t2 uint16_t
#define PI 3.14159265359

extern "C" __declspec(dllexport) void OCT_FFT(data_t2 * h_data, data_t2 * h_processed_data, data_t * dispersion, data_t * hann, uint32_t z0, uint32_t DEPTH, uint32_t NALINES, uint32_t SAMPLES);
int main(int argc, char** argv) {
	data_t2* h_data;
	data_t2* h_processed_data;
	FILE* fp;
	long long int i, j;
	time_t start_time, end_time;

	uint32_t signal_f = 100;
	uint32_t NALINES = 1000*1000;
	uint32_t SAMPLES = 2048;
	uint32_t DEPTH = 200;

	long long int length_spectral = (long long int)SAMPLES * (long long int)NALINES * (long long int)1;   // channel number = 1
	long long int length_spatial = (long long int)DEPTH * (long long int)NALINES * (long long int)1; // channel number = 1, calculate only amplitude

	h_data = (data_t2*)malloc(sizeof(data_t2) * (long long int)length_spectral);   
	h_processed_data = (data_t2*)malloc(sizeof(data_t2) * (long long int)length_spatial);  

	data_t* hann;
	hann = (data_t*)malloc(sizeof(data_t) * SAMPLES);
	for (i = 0; i < SAMPLES; i++) {
		hann[i] = 0.5 * (1 - cos(2 * PI * i / (SAMPLES - 1)));
	}

	data_t* dispersion;
	dispersion = (data_t*)malloc(sizeof(data_t) * SAMPLES * 2);
	for (i = 0; i < SAMPLES; i++) {
		dispersion[2 * i] = 1;
		dispersion[2 * i + 1] = 0;
	}

	// init raw data
	for (j = 0; j < SAMPLES; j++) {
		h_data[j] = (data_t2)((sin(2 * PI * signal_f * j / SAMPLES + PI / 6) + 1) / 2 * 65535);
	}
	for (i = 1; i < NALINES; i++) {
		for (j = 0; j < SAMPLES; j++) {
			h_data[i * SAMPLES + j] = h_data[j];
		}
		
	}
	//printf("finished initializing data, size of raw data: %lld\n", (long long int)length_spectral* sizeof(data_t2));
	printf("start processing...\n");
	
	// run 20 times, calculate average GPU time
	time(&start_time);
	for (i = 0; i < 1; i++) {
		OCT_FFT(h_data, h_processed_data, dispersion, hann, 0, DEPTH, NALINES, SAMPLES);
	}
	time(&end_time);
	printf("time per run: %.2f seconds\n", difftime(end_time, start_time)/1);

	printf("finished processing...\n");
	//printf("finished processing data, size of processed data: %lld\n", (long long int)length(h_processed_data));
	fp = fopen("GPU_h_processed.txt", "w");
	if (fp != NULL) {
		for (i = 0; i < 20; i++) {
			for (j = 0; j < DEPTH; j++) {
				fprintf(fp, "%d ", h_processed_data[i * DEPTH + j]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		printf("write to file success\n");
	}
	else
		printf("open file failed\n");

	return 0;
}


void OCT_FFT(data_t2* h_data, data_t2* h_processed_data, data_t* dispersion, data_t* hann, uint32_t z0, uint32_t DEPTH, uint32_t NALINES, uint32_t SAMPLES) {
	// h_data: OCT spectral data in computer memory
	// h_processed_data: OCT spatial data location in computer memory
	// dispersion: dispersion array, complex array in float format, dispersion[0] is real value of first element, dispersion[1] is imag value of first element
	// hann: hanning window array
	// z0: start depth to save
	// DEPTH: number of depth pixels to save
	// NALINES: total number of Alines to process
	// SAMPLES: number of samples per Aline

	long long int FFT_samples;
	FFT_samples = 2;
	while (FFT_samples < SAMPLES) {
		FFT_samples = FFT_samples * 2;
	}
	//printf("FFT length: %d \n", FFT_samples);

	// declear FFT function
	void my_cufft(data_t2* h_data, data_t* d_dispersion, data_t* d_hann, data_t2* h_processed_data, uint32_t z0, uint32_t DEPTH, uint32_t NALINES, uint32_t SAMPLES, uint32_t FFT_samples);

	//init GPU memories
	data_t2* d_data;
	data_t2* d_processed_data;
	data_t* d_dispersion;
	data_t* d_hann;

	//pre-allocate GPU memory
	checkCudaErrors(cudaMalloc((void**)&d_data, (long long int)NALINES * (long long int)sizeof(data_t2) * (long long int)SAMPLES));
	checkCudaErrors(cudaMalloc((void**)&d_processed_data, (long long int)NALINES * sizeof(data_t2) * DEPTH));
	checkCudaErrors(cudaMalloc((void**)&d_dispersion, sizeof(data_t) * SAMPLES * 2));
	checkCudaErrors(cudaMalloc((void**)&d_hann, sizeof(data_t) * SAMPLES));

	// transfer spectral data, dispersion, window array from computer memory to GPU memory
	checkCudaErrors(cudaMemcpy(d_dispersion, dispersion, sizeof(data_t) * SAMPLES * 2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hann, hann, sizeof(data_t) * SAMPLES, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_data, h_data, (long long int)NALINES * sizeof(data_t2) * SAMPLES, cudaMemcpyHostToDevice));

	//perform GPU FFT
	my_cufft(d_data, d_dispersion, d_hann, d_processed_data, z0, DEPTH, NALINES, SAMPLES, FFT_samples);

	// transfer post-FFT data back to computer memory
	checkCudaErrors(cudaMemcpy(h_processed_data, d_processed_data, (long long int)NALINES * sizeof(data_t2) * DEPTH, cudaMemcpyDeviceToHost));
	
	//free GPU memory
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_processed_data));
	checkCudaErrors(cudaFree(d_dispersion));
	checkCudaErrors(cudaFree(d_hann));
	cudaDeviceReset();
	
}


static __global__ void copy_data(long long int length_fft_data, data_t* FFT_buffer, data_t2* d_data, data_t* dispersion, data_t* hann, uint32_t SAMPLES, uint32_t FFT_samples) {
	//copy data to compex FFT buffer
	const int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	const int threadID = blockID * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	//cuPrintf("blockID: %d, threadID: %d, numTHreads: %d, datanum:%d\n", blockID, threadID, numThreads, length_fft_data);
	long long int i;
	// copy d_data into FFT_buffer
	for (i = threadID; i < length_fft_data; i += numThreads) {
		// map d_data index into FFT_buffer index
		long long int k = (i % SAMPLES) * 2 + long long int(i / SAMPLES) * FFT_samples * 2;
		// copy data into FFT_buffer, rescale to plus-minus 0.2V scale
		data_t tmp = ((data_t)d_data[i] * 0.4 / pow(2, sizeof(data_t2)) - 0.2);
		FFT_buffer[k] =  tmp * hann[i % SAMPLES] * dispersion[i % SAMPLES * 2];
		FFT_buffer[k+1] = tmp * hann[i % SAMPLES] * dispersion[i % SAMPLES * 2 + 1];
		//cuPrintf("FFT_buffer %d is %f\n", i + (long long int)part*length_fft_data, FFT_buffer[k]);
	}

}

static __global__ void my_hilbert(data_t* FFT_buffer, long long int length_fft) {
	const int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	const int threadID = blockID * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

	long long int i;
	for (i = threadID; i < long long int(length_fft / 2); i += numThreads) {
		//new_data[i % 1440*2 + i / 1440 * 4096] = old_data[i]
		long long int k = (i % 2048) + long long int(i / 2048) * 4096;
		if (i % 2048 > 1) {
			FFT_buffer[k] = 2 * FFT_buffer[k] / 2048;
			FFT_buffer[k + 2048] = 0;
			//cuPrintf("FFT_buffer is %f, %f\n", FFT_buffer[k], FFT_buffer[k + 2048]);
		}
		else {
			FFT_buffer[k] = FFT_buffer[k] / 2048;
			FFT_buffer[k + 2048] = 0;
		}
	}
}

static __global__ void my_dispersion(data_t* FFT_buffer, data_t* dispersion, long long int length_fft_data) {
	const int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	const int threadID = blockID * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	long long int i;
	for (i = threadID; i < length_fft_data; i += numThreads) {
		long long int k = (i % 1440) * 2 + long long int(i / 1440) * 4096;
		int d = (i % 1440) * 2;
		//dispersion compensation
		data_t a = FFT_buffer[k];
		data_t b = FFT_buffer[k + 1];
		FFT_buffer[k] = (a * dispersion[d] - b * dispersion[d + 1]);
		FFT_buffer[k + 1] = (a * dispersion[d + 1] + b * dispersion[d]);
		//cuPrintf("FFT_buffer %d is %.10f, %.10f\n", k, FFT_buffer[k], FFT_buffer[k+1]);
	}
}

static __global__ void calc_amp(data_t* FFT_buffer, data_t2* d_processed_data, uint32_t z0, uint32_t DEPTH, uint32_t NALINES, uint32_t SAMPLES, uint32_t FFT_samples) {
	const int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	const int threadID = blockID * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	long long int data_length;
	data_length = DEPTH * NALINES;
	long long int i;
	for (i = threadID; i < data_length; i += numThreads) {
		// map d_processed_data index into FFT_buffer index
		long long int d = long long int(i / DEPTH) * FFT_samples * 2 + (i % DEPTH + z0) * 2;
		data_t a = (data_t)pow(FFT_buffer[d], 2) + pow(FFT_buffer[d + 1], 2);
		d_processed_data[i] = data_t2(sqrt(a) * pow(2, sizeof(data_t2)));
	}
}

void my_cufft(uint16_t* d_data, data_t* dispersion, data_t* hann, data_t2* d_processed_data, uint32_t z0, uint32_t DEPTH, uint32_t NALINES, uint32_t SAMPLES, uint32_t FFT_samples) {
	// d_data: spectral data in GPU memory, multiple Alines included
	// dispersion: dispersion array in GPU memory, complex array stored in float format, dispersion[0] is real value of first element, dispersion[1] is imag value of first element
	// hann: hanning window array in GPU memory
	// d_processed_data: spatial data buffer in GPU memory
	// z0: start depth to save
	// DEPTH: number of depth pixels to save
	// NALINES: total number of Alines to process
	// SAMPLES: number of samples per Aline
	// FFT_samples, number of samples per FFT, should be 2^X

	long long int length_fft, length_fft_data;
    // length of FFT results for all Alines, include real and imag parts
	length_fft = (long long int) NALINES * FFT_samples * 2;
	// length of data going to perform FFT
	length_fft_data = (long long int) NALINES * SAMPLES;
	// Init FFT function
	cufftHandle plan;
	cufftPlan1d(&plan, FFT_samples, CUFFT_C2C, (long long int) NALINES); // deprecated?
	// Init FFT buffer, complex array assembled in float format, FFT_buffer[0] is real value of first element, FFT_buffer[1] is imag value of first element
	data_t* FFT_buffer;
	checkCudaErrors(cudaMalloc((void**)&FFT_buffer, sizeof(data_t) * length_fft));
	// configure GPU CUDA core configuration
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(16, 16, 1);

	//copy real value data to complex FFT buffer
	cudaMemset(FFT_buffer, 0, length_fft * (long long int)sizeof(data_t));
	copy_data << <dimBlock, dimGrid >> > (length_fft_data, FFT_buffer, d_data, dispersion, hann, SAMPLES, FFT_samples);

	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	//CUDAFFT
	checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(FFT_buffer), reinterpret_cast<cufftComplex*>(FFT_buffer), CUFFT_FORWARD));

	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	//my_hilbert << <dimBlock, dimGrid>> > ((data_t *)FFT_buffer, length_fft);
	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	//printf("finished hilbert\n");
	// checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(FFT_buffer), reinterpret_cast<cufftComplex *>(FFT_buffer), CUFFT_INVERSE));
	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	//printf_data << <a, b >> > (dispersion, length_fft);
	//dispersion compensation
	//my_dispersion << <dimBlock, dimGrid>> > ((data_t *)FFT_buffer, dispersion, length_fft_data);
	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	//to space domain
	//checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(FFT_buffer), reinterpret_cast<cufftComplex *>(FFT_buffer), CUFFT_FORWARD));
	//printf_data << <a, b >> > (FFT_buffer, length_fft);
	calc_amp << <dimBlock, dimGrid >> > ((data_t*)FFT_buffer, d_processed_data, z0, DEPTH, NALINES, SAMPLES, FFT_samples);
	//checkCudaErrors(cudaMemcpy(h_FFT_buffer, FFT_buffer, length_fft * sizeof(data_t),
	//	cudaMemcpyDeviceToHost));
	//printf_data << <a, b >> > (d_processed, length_fft);

	checkCudaErrors(cudaFree(FFT_buffer));
	//checkCudaErrors(cudaFree(plan));
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
