//process single channel of PSOCT and outputs amplitdue and phase

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
//#include <time.h>
#include <math.h>
//#include "cuPrintf.cu"
//#include "cuPrintf.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#define data_t float
#define data_t2 uint16_t
#define PI 3.14159265359
#define sample_f 1440
#define signal_f 100
#define PRINT_TIME 1

extern "C" __declspec(dllexport) void PSOCT_process(data_t2 *h_signal, data_t2 *h_processed_signal, int z0, int DEPTH, int a, int b, int NALINES, int NBLINES, int SAMPLES);
int main(int argc, char **argv) {
	data_t2 *h_signal;
	data_t2 *h_processed_signal;
    FILE *fp;
    long long int i;
	
	int NALINES = 1250;
	int NBLINES = 200;
	int SAMPLES = 1440;
	int DEPTH = 200;

    long long int length_raw = (long long int)SAMPLES * (long long int)NALINES * (long long int)NBLINES * (long long int)1;  
    long long int length_data = (long long int)DEPTH * (long long int)2 * (long long int)NALINES * (long long int)NBLINES;

    h_signal = (data_t2 *)malloc(sizeof(data_t2) * (long long int)length_raw);   //1440*1100*1250*2

    h_processed_signal = (data_t2 *)malloc(sizeof(data_t2) * (long long int)length_data);  //600*1100*1250
	
    // init raw data
    for (i = 0; i < length_raw; i++) {
      h_signal[i] = (data_t2)((sin(2*PI*signal_f*i/sample_f)+1)/2*65535);
    }
	
    printf("start processing...\n");
	PSOCT_process(h_signal, h_processed_signal, 0,DEPTH,280,1100,NALINES, NBLINES, SAMPLES);
	printf("finished processing...\n");
	printf("%d, %d, %d, %d\n", h_processed_signal[0], h_processed_signal[1], h_processed_signal[2], h_processed_signal[3]);

	fp = fopen("GPU_h_processed.txt", "w");
	if (fp != NULL) {
		for (i = 0; i < NALINES * DEPTH * 20; i += 4)
			fprintf(fp, "%d,%d,%d,%d\n", h_processed_signal[i], h_processed_signal[i + 1], h_processed_signal[i + 2], h_processed_signal[i + 3]);
		//printf("%d,%d,%d,%d\n", h_processed_signal[i], h_processed_signal[i + 1], h_processed_signal[i + 2], h_processed_signal[i + 3]);
		fclose(fp);
		printf("write to file success\n");
	}
	else
		printf("open file failed\n");
return 0;
}


void PSOCT_process(data_t2 *h_signal, data_t2 *h_processed_signal, int z0, int DEPTH, int a, int b, int NALINES, int NBLINES, int SAMPLES) {
	void my_cufft(data_t2 *d_signal, data_t *dispersion, data_t *hann, data_t2 *d_processed_amp, data_t2 *d_processed_phase, int z0, int DEPTH, int numBline, int NALINES);
    long long int i;
	//3x3mm FOV with 3um step size is much more data than the GPU memory, we need to process them separately
	int *size;
	size = (int *)malloc(sizeof(int) * 10);
	int n;
	if (NALINES <= 1400) {
		for (n = 1; n<int(NBLINES / 600)+1; n++) {
			size[n] = 600;
		}
		size[n] = NBLINES - 600 * (n - 1);
		//printf("size : %d\n", size[n]);
	}
	else {
		for (n = 1; n<int(NBLINES / 400)+1; n++) {
			size[n] = 400;
		}
		size[n] = NBLINES - 400 * (n - 1);
	}
    data_t *dispersion, *d_dispersion, *hann, *d_hann;
    data_t f, arg;
    data_t k0,k1,dk,kc;
    data_t2   *d_processed_amp, *d_processed_phase;
	data_t2 *d_dataA;
    dispersion = (data_t *) malloc(sizeof(data_t) * SAMPLES * 2);
    //init dispersion compensation array
	k0 = 2 * PI / 1363 ;
	k1 = 2 * PI / 1227 ;
	kc = 2 * PI / 1295 ;
	dk = (k1 - k0) / (SAMPLES-1);
    a = data_t(a * pow(10, 4));
    b = data_t(b * pow(10, 6));

    for (i = 0; i < SAMPLES; i ++) {
        f = 3 * (k0 - kc + (i) * dk);
        arg = a * pow(f, 2) + b * pow(f, 3);
        dispersion[2*i] = (data_t) cos(arg);
        dispersion[2*i + 1] = (data_t) sin(arg);
    }

	//init hann window array
	hann= (data_t *)malloc(sizeof(data_t) * SAMPLES);
	for (i = 0; i < SAMPLES; i++) {
		hann[i] = 0.5*(1 - cos(2 * PI*i / (SAMPLES-1)));
		//printf("%.10f\n", hann[i]);
	}


	//pre-allocate GPU memory
	checkCudaErrors(cudaMalloc((void **) &d_dataA, (long long int)NALINES * (long long int)size[1] * (long long int)sizeof(data_t2) * (long long int)SAMPLES));
	checkCudaErrors(cudaMalloc((void **) &d_processed_amp, (long long int)NALINES * size[1] * sizeof(data_t2) * DEPTH));
	checkCudaErrors(cudaMalloc((void **)&d_processed_phase, (long long int)NALINES * size[1] * sizeof(data_t2) * DEPTH));
	checkCudaErrors(cudaMalloc((void **) &d_dispersion, sizeof(data_t) * SAMPLES * 2));
	checkCudaErrors(cudaMalloc((void **)&d_hann, sizeof(data_t) * SAMPLES));

	checkCudaErrors(cudaMemcpy(d_dispersion, dispersion, sizeof(data_t) * SAMPLES * 2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hann, hann, sizeof(data_t) * SAMPLES, cudaMemcpyHostToDevice));

	//GPU FFT
	for (int iter = 1; iter <= n; iter++) {
		//previous processed data size
		int pre_data;
		pre_data = 0;
		for (int pre = iter - 1; pre > 0; pre--) {
			pre_data = pre_data + size[pre];
		}
		//printf("size: %d\n", size[iter]);
		//printf("pre_data: %d\n", pre_data);
		checkCudaErrors(cudaMemcpy(d_dataA, h_signal + (long long int)NALINES*pre_data*SAMPLES, (long long int)NALINES * size[iter] * sizeof(data_t2) * SAMPLES, cudaMemcpyHostToDevice));
		my_cufft(d_dataA, d_dispersion, d_hann, d_processed_amp, d_processed_phase, z0, DEPTH, size[iter], NALINES);
		checkCudaErrors(cudaMemcpy(h_processed_signal + (long long int)NALINES *pre_data * DEPTH, d_processed_amp, (long long int)NALINES * size[iter] * sizeof(data_t2) * DEPTH, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_processed_signal + (long long int)NALINES *(pre_data + NBLINES) * DEPTH, d_processed_phase, (long long int)NALINES * size[iter] * sizeof(data_t2) * DEPTH, cudaMemcpyDeviceToHost));
		//printf("finished memcopy device to host for first half\n");
	}
	//free GPU memory
	checkCudaErrors(cudaFree(d_dataA));
	checkCudaErrors(cudaFree(d_processed_amp));
	checkCudaErrors(cudaFree(d_processed_phase));
	checkCudaErrors(cudaFree(d_dispersion));
	checkCudaErrors(cudaFree(d_hann));
    cudaDeviceReset();
}


static __global__ void copy_data(long long int length_fft_data, int part, data_t *FFT_buffer, uint16_t *d_signal, data_t *hann) {
	//copy data to compex FFT buffer
	const int blockID = blockIdx.x + blockIdx.y*gridDim.x;
	const int threadID = blockID * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	//cuPrintf("blockID: %d, threadID: %d, numTHreads: %d, datanum:%d\n", blockID, threadID, numThreads, length_fft_data);
	long long int i;
	for (i = threadID; i < length_fft_data; i += numThreads) {
		long long int k = (i % 1440) * 2 + long long int(i / 1440) * 4096;
	    FFT_buffer[k] = (d_signal[i + (long long int)part*length_fft_data]*0.4/65535-0.2)*hann[i % 1440];
		//cuPrintf("FFT_buffer %d is %f\n", i + (long long int)part*length_fft_data, FFT_buffer[k]);
	}
	
}

static __global__ void my_hilbert(data_t *FFT_buffer, long long int length_fft) {
	const int blockID = blockIdx.x + blockIdx.y*gridDim.x;
	const int threadID = blockID * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

	long long int i;
	for (i = threadID; i < long long int(length_fft / 2); i += numThreads) {
		//new_data[i % 1440*2 + i / 1440 * 4096] = old_data[i]
		long long int k =( i % 2048) + long long int(i / 2048) * 4096;
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

static __global__ void my_dispersion(data_t *FFT_buffer, data_t *dispersion, long long int length_fft_data) {
	const int blockID = blockIdx.x + blockIdx.y*gridDim.x;
	const int threadID = blockID * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	long long int i;
	for (i = threadID; i < length_fft_data; i += numThreads) {
		long long int k = (i % 1440) * 2 + long long int(i / 1440) * 4096;
		int d = (i % 1440) * 2;
	//dispersion compensation
		data_t a = FFT_buffer[k];
		data_t b = FFT_buffer[k+1];
		FFT_buffer[k] = (a * dispersion[d] - b * dispersion[d + 1]);
		FFT_buffer[k + 1] = (a * dispersion[d + 1] + b * dispersion[d]);
		//cuPrintf("FFT_buffer %d is %.10f, %.10f\n", k, FFT_buffer[k], FFT_buffer[k+1]);
	}
}

static __global__ void calc_amp(data_t *FFT_buffer, int num_Blines_per_FFT, int part, data_t2 *d_processed_amp, data_t2 *d_processed_phase, int z0, int DEPTH, int NALINES) {
	const int blockID = blockIdx.x + blockIdx.y*gridDim.x;
	const int threadID = blockID * (blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

	long long int i;
	long long range = (long long int)num_Blines_per_FFT*NALINES * DEPTH;
	long long int step = (long long int)part * num_Blines_per_FFT*NALINES*DEPTH;
	for (i = threadID; i < range; i += numThreads) {
		long long int k = i;   //
		long long int d = long long int(i / DEPTH) *4096 + (i % DEPTH + z0) * 2;
		data_t a = (data_t)sqrt(pow(FFT_buffer[d], 2) + pow(FFT_buffer[d + 1], 2));
		int sign_a_real = (FFT_buffer[d] > 0) - (FFT_buffer[d] < 0);
		int sign_a_imag = (FFT_buffer[d + 1] > 0) - (FFT_buffer[d + 1] < 0);
		data_t p = (data_t)atan(FFT_buffer[d + 1] / FFT_buffer[d])+PI/2*sign_a_imag-PI/2*sign_a_imag*sign_a_real;
		
		//if (FFT_buffer[d] < 0.001) {
		//	p = 0;
		//}
		d_processed_amp[k + step] = data_t2(a / 4 * 65535);;
		d_processed_phase[k + step] = data_t2((p +PI) / 2/PI * 65535);
	}
}

void my_cufft(data_t2 *d_signal, data_t *dispersion, data_t *hann, data_t2 *d_processed_amp, data_t2 *d_processed_phase, int z0, int DEPTH, int numBline, int NALINES) {
    int num_Blines_per_FFT;
    int part;
    long long int length_fft, length_fft_data;
    num_Blines_per_FFT=20;
    length_fft=(long long int)num_Blines_per_FFT*NALINES*4096;
    length_fft_data=(long long int)num_Blines_per_FFT*NALINES*1440;
	data_t *FFT_buffer;
	cufftHandle plan;
	cufftPlan1d(&plan, 2048, CUFFT_C2C, (long long int)NALINES * num_Blines_per_FFT);
	checkCudaErrors(cudaMalloc((void **) &FFT_buffer, sizeof(data_t) * length_fft));
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(20, 20, 1);
    

    for(part=0; part< numBline/num_Blines_per_FFT; part++){
		//printf("part= %d   ,length_fft=%lld, length_fft_data=%lld , %ld\n", part, length_fft, length_fft_data, plan);
        //copy data to compex FFT buffer
		cudaMemset(FFT_buffer, 0, length_fft * (long long int)sizeof(data_t));
		copy_data << <dimBlock , dimGrid>> > (length_fft_data, part, FFT_buffer, d_signal, hann);

		//printf_data << <a, b >> > (FFT_buffer, length_fft);
		//printf_data << <a, b >> > (FFT_buffer, length_fft);
        //Hilbert transform
        checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(FFT_buffer), reinterpret_cast<cufftComplex *>(FFT_buffer), CUFFT_FORWARD));
        
		//printf_data << <a, b >> > (FFT_buffer, length_fft);
		//my_hilbert << <dimBlock, dimGrid>> > ((data_t *)FFT_buffer, length_fft);
		//printf_data << <a, b >> > (FFT_buffer, length_fft);
		//printf("finished hilbert\n");
        //checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(FFT_buffer), reinterpret_cast<cufftComplex *>(FFT_buffer), CUFFT_INVERSE));
		//printf_data << <a, b >> > (FFT_buffer, length_fft);
		//printf_data << <a, b >> > (dispersion, length_fft);
        //dispersion compensation
		//my_dispersion << <dimBlock, dimGrid>> > ((data_t *)FFT_buffer, dispersion, length_fft_data);
		//printf_data << <a, b >> > (FFT_buffer, length_fft);
        //to space domain
        //checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(FFT_buffer), reinterpret_cast<cufftComplex *>(FFT_buffer), CUFFT_FORWARD));
		//printf_data << <a, b >> > (FFT_buffer, length_fft);
		calc_amp << <dimBlock, dimGrid>> > ((data_t *)FFT_buffer, num_Blines_per_FFT, part, d_processed_amp, d_processed_phase, z0, DEPTH, NALINES);
		//checkCudaErrors(cudaMemcpy(h_FFT_buffer, FFT_buffer, length_fft * sizeof(data_t),
		//	cudaMemcpyDeviceToHost));
		//printf_data << <a, b >> > (d_processed, length_fft);
    }
	
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
