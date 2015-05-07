/*
 * GpuRSA.cpp
 *
 *  Created on: 2015-4-23
 *      Author: Yuqing Guan
 */

#include "GpuRSA.h"

namespace RSA
{

__constant__ int d_SMALL_PRIMES[SMALL_PRIMES_COUNT];
__constant__ unsigned long long d_R[32];

/**
 * Init curand
 */
__global__ void gpuRandInit(GpuRSA *rsa)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock(), tid, 0, rsa->states + tid);
}

GpuRSA::GpuRSA()
{
	deviceReady = false;

	int count;
	GPU_CHECKERROR(cudaGetDeviceCount(&count));

	if (count > 0)
	{
		deviceReady = true;
	}
	else
	{
		return;
	}

	cudaDeviceProp prop;
	GPU_CHECKERROR(cudaGetDeviceProperties(&prop, 0));

	maxBlockX = prop.maxGridSize[0];

	unsigned long long R[32];

	memset(R, 0, arrSize);
	R[16] = 1;

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_SMALL_PRIMES, SMALL_PRIMES,
					sizeof(SMALL_PRIMES)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_R, R, arrSize));

	GPU_CHECKERROR(cudaMalloc(&E, arrSize));
	GPU_CHECKERROR(cudaMalloc(&D, arrSize));
	GPU_CHECKERROR(cudaMalloc(&N, arrSize));

	GPU_CHECKERROR(cudaMalloc(&states, 1024 * sizeof(curandState)));

	GPU_CHECKERROR(cudaMalloc(&d_this, sizeof(GpuRSA)));

	GPU_CHECKERROR(
			cudaMemcpy(d_this, this, sizeof(GpuRSA), cudaMemcpyHostToDevice));

	gpuRandInit<<<32, 32>>>(d_this);
}

GpuRSA::~GpuRSA()
{
	GPU_CHECKERROR(cudaFree(E));
	GPU_CHECKERROR(cudaFree(D));
	GPU_CHECKERROR(cudaFree(N));

	GPU_CHECKERROR(cudaFree(states));

	GPU_CHECKERROR(cudaFree(d_this));
}

/**
 * m ^ e % N = c
 */
__global__ void gpuEncrypt(char *in, char *out, GpuRSA *rsa)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	__shared__ char cBuffer[256];

	int bufferBase = tx * 4;
	int inBase = bufferBase + bx * 127;
	int outBase = inBase + bx;

	for (int i = 0; i < 3; ++i)
	{
		cBuffer[bufferBase + i] = in[inBase + i];
		cBuffer[bufferBase + 128 + i] = 0;
	}

	cBuffer[bufferBase + 3] = tx < 31 ? in[inBase + 3] : 0;

	__syncthreads();

	unsigned long long *buffer = (unsigned long long *) cBuffer;

	__shared__ unsigned long long invR[32], invP[32];

	rsa->extGcd(d_R, rsa->N, invR);
	rsa->extGcd(rsa->N, d_R, invP);

	rsa->powMod(buffer, rsa->E, rsa->N, buffer, invR, invP);

	for (int i = 0; i < 4; ++i)
	{
		out[outBase + i] = cBuffer[bufferBase + i];
	}
}

/**
 * c ^ d % N = m
 */
__global__ void gpuDecrypt(char *in, char *out, GpuRSA *rsa)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	__shared__ char cBuffer[256];

	int bufferBase = tx * 4;
	int inBase = bufferBase + bx * 128;
	int outBase = inBase - bx;

	for (int i = 0; i < 4; ++i)
	{
		cBuffer[bufferBase + i] = in[inBase + i];
		cBuffer[bufferBase + 128 + i] = 0;
	}

	unsigned long long *buffer = (unsigned long long *) cBuffer;

	__shared__ unsigned long long invR[32], invP[32];

	rsa->extGcd(d_R, rsa->N, invR);
	rsa->extGcd(rsa->N, d_R, invP);

	rsa->powMod(buffer, rsa->D, rsa->N, buffer, invR, invP);

	for (int i = 0; i < 3; ++i)
	{
		out[outBase + i] = cBuffer[bufferBase + i];
	}

	if (tx < 31)
	{
		out[outBase + 3] = cBuffer[bufferBase + 3];
	}
}

float GpuRSA::encryptFile(const char *inName, const char *outName)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	if (!publicKeyReady)
	{
		return KEY_ERROR;
	}

	FILE *inFile, *outFile;

	inFile = fopen(inName, "rb");
	if (inFile == NULL)
	{
		return IN_FILE_ERROR;
	}

	outFile = fopen(outName, "wb");
	if (outFile == NULL)
	{
		fclose(inFile);

		return OUT_FILE_ERROR;
	}

	fseek(inFile, 0, SEEK_END);
	size_t size = ftell(inFile);

	float gpuTime;
	cudaEvent_t gpuStart, gpuEnd;

	GPU_CHECKERROR(cudaEventCreate(&gpuStart));
	GPU_CHECKERROR(cudaEventCreate(&gpuEnd));

	GPU_CHECKERROR(cudaEventRecord(gpuStart, 0));

	fseek(inFile, 0, SEEK_SET);

	size_t pos = 0;
	int runningStreams;

	int *currentInChunkSize = new int[maxStoredChunks];
	int *currentOutChunkSize = new int[maxStoredChunks];

	cudaStream_t *stream = new cudaStream_t[maxStoredChunks];

	char **inBuffer = new char*[maxStoredChunks];
	char **outBuffer = new char*[maxStoredChunks];

	int times = chunkSize / 127;

	char **d_In = new char*[maxStoredChunks];
	char **d_Out = new char*[maxStoredChunks];

	dim3 *gridDim = new dim3[maxStoredChunks];
	dim3 *blockDim = new dim3[maxStoredChunks];

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		GPU_CHECKERROR(cudaStreamCreate(stream + i));
		GPU_CHECKERROR(
				cudaHostAlloc(inBuffer + i, chunkSize + 127,
						cudaHostAllocDefault));
		GPU_CHECKERROR(
				cudaHostAlloc(outBuffer + i, chunkSize + 128 + times,
						cudaHostAllocDefault));
		GPU_CHECKERROR(cudaMalloc(d_In + i, chunkSize + 127));
		GPU_CHECKERROR(cudaMalloc(d_Out + i, chunkSize + 128 + times));
	}

	int blockX = min(times, maxBlockX);
	int blockY = (int) ceil(times / (double) maxBlockX);

	dim3 defaultGridDim(blockX, blockY, 1);
	dim3 defaultBlockDim(32, 1, 1);

	while (pos < size)
	{
		runningStreams = 0;

		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentInChunkSize[i] = chunkSize;
			if (pos + currentInChunkSize[i] > size)
			{
				currentInChunkSize[i] = size - pos;
			}

			if (fread(inBuffer[i], 1, currentInChunkSize[i], inFile)
					!= currentInChunkSize[i])
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentInChunkSize[i];
			++runningStreams;

			times = currentInChunkSize[i] / 127 + (pos == size);
			int newLength = times * 127;
			int padding = newLength - currentInChunkSize[i];

			GPU_CHECKERROR(
					cudaMemcpyAsync(d_In[i], inBuffer[i], currentInChunkSize[i],
							cudaMemcpyHostToDevice, stream[i]));

			if (pos == size)
			{
				GPU_CHECKERROR(
						cudaMemsetAsync(d_In[i] + currentInChunkSize[i],
								padding, padding, stream[i]));
				currentInChunkSize[i] += padding;

				blockX = min(times, maxBlockX);
				blockY = (int) ceil(times / (double) maxBlockX);

				gridDim[i] = dim3(blockX, blockY, 1);
				blockDim[i] = dim3(32, 1, 1);
			}
			else
			{
				gridDim[i] = defaultGridDim;
				blockDim[i] = defaultBlockDim;
			}

			currentOutChunkSize[i] = times * 128;
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			gpuEncrypt<<<gridDim[i], blockDim[i], 0, stream[i]>>>(d_In[i],
					d_Out[i], d_this);
			GPU_CHECKERROR(cudaGetLastError());
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(
					cudaMemcpyAsync(outBuffer[i], d_Out[i],
							currentOutChunkSize[i], cudaMemcpyDeviceToHost,
							stream[i]));
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(cudaStreamSynchronize(stream[i]));

			if (fwrite(outBuffer[i], 1, currentOutChunkSize[i], outFile)
					!= currentOutChunkSize[i])
			{
				fclose(inFile);
				fclose(outFile);

				return OUT_FILE_ERROR;
			}
		}
	}

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		GPU_CHECKERROR(cudaStreamDestroy(stream[i]));
		GPU_CHECKERROR(cudaFreeHost(inBuffer[i]));
		GPU_CHECKERROR(cudaFreeHost(outBuffer[i]));
		GPU_CHECKERROR(cudaFree(d_In[i]));
		GPU_CHECKERROR(cudaFree(d_Out[i]));
	}

	delete gridDim;
	delete blockDim;

	delete stream;
	delete currentInChunkSize;
	delete currentOutChunkSize;

	delete d_In;
	delete d_Out;

	delete inBuffer;

	fclose(inFile);
	fclose(outFile);

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

float GpuRSA::decryptFile(const char *inName, const char *outName)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	if (!privateKeyReady)
	{
		return KEY_ERROR;
	}

	FILE *inFile, *outFile;

	inFile = fopen(inName, "rb");
	if (inFile == NULL)
	{
		return IN_FILE_ERROR;
	}

	outFile = fopen(outName, "wb");
	if (outFile == NULL)
	{
		fclose(inFile);

		return OUT_FILE_ERROR;
	}

	fseek(inFile, 0, SEEK_END);
	size_t size = ftell(inFile);

	if ((size & 15) != 0)
	{
		fclose(inFile);
		fclose(outFile);

		return TEXT_ERROR;
	}

	float gpuTime;
	cudaEvent_t gpuStart, gpuEnd;

	GPU_CHECKERROR(cudaEventCreate(&gpuStart));
	GPU_CHECKERROR(cudaEventCreate(&gpuEnd));

	GPU_CHECKERROR(cudaEventRecord(gpuStart, 0));

	fseek(inFile, 0, SEEK_SET);

	size_t pos = 0;
	int runningStreams;

	int *currentInChunkSize = new int[maxStoredChunks];
	int *currentOutChunkSize = new int[maxStoredChunks];

	cudaStream_t *stream = new cudaStream_t[maxStoredChunks];

	char **inBuffer = new char*[maxStoredChunks];
	char **outBuffer = new char*[maxStoredChunks];

	int times = chunkSize / 127;

	char **d_In = new char*[maxStoredChunks];
	char **d_Out = new char*[maxStoredChunks];

	dim3 *gridDim = new dim3[maxStoredChunks];
	dim3 *blockDim = new dim3[maxStoredChunks];

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		GPU_CHECKERROR(cudaStreamCreate(stream + i));
		GPU_CHECKERROR(
				cudaHostAlloc(inBuffer + i, chunkSize + times,
						cudaHostAllocDefault));
		GPU_CHECKERROR(
				cudaHostAlloc(outBuffer + i, chunkSize, cudaHostAllocDefault));
		GPU_CHECKERROR(cudaMalloc(d_In + i, chunkSize + times));
		GPU_CHECKERROR(cudaMalloc(d_Out + i, chunkSize));
	}

	int blockX = min(times, maxBlockX);
	int blockY = (int) ceil(times / (double) maxBlockX);

	dim3 defaultGridDim(blockX, blockY, 1);
	dim3 defaultBlockDim(32, 1, 1);

	while (pos < size)
	{
		runningStreams = 0;

		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentInChunkSize[i] = chunkSize + times;
			if (pos + currentInChunkSize[i] > size)
			{
				currentInChunkSize[i] = size - pos;
			}

			if (fread(inBuffer[i], 1, currentInChunkSize[i], inFile)
					!= currentInChunkSize[i])
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentInChunkSize[i];
			++runningStreams;

			times = currentInChunkSize[i] / 128;

			GPU_CHECKERROR(
					cudaMemcpyAsync(d_In[i], inBuffer[i], currentInChunkSize[i],
							cudaMemcpyHostToDevice, stream[i]));

			if (pos == size)
			{
				blockX = min(times, maxBlockX);
				blockY = (int) ceil(times / (double) maxBlockX);

				gridDim[i] = dim3(blockX, blockY, 1);
				blockDim[i] = dim3(32, 1, 1);
			}
			else
			{
				gridDim[i] = defaultGridDim;
				blockDim[i] = defaultBlockDim;
			}

			currentOutChunkSize[i] = times * 127;
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			gpuDecrypt<<<gridDim[i], blockDim[i], 0, stream[i]>>>(d_In[i],
					d_Out[i], d_this);
			GPU_CHECKERROR(cudaGetLastError());
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(
					cudaMemcpyAsync(outBuffer[i], d_Out[i],
							currentOutChunkSize[i], cudaMemcpyDeviceToHost,
							stream[i]));
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(cudaStreamSynchronize(stream[i]));

			if (pos == size && i == runningStreams - 1)
			{
				currentOutChunkSize[i] -= outBuffer[i][currentOutChunkSize[i]
						- 1];
			}

			if (fwrite(outBuffer[i], 1, currentOutChunkSize[i], outFile)
					!= currentOutChunkSize[i])
			{
				fclose(inFile);
				fclose(outFile);

				return OUT_FILE_ERROR;
			}
		}
	}

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		GPU_CHECKERROR(cudaStreamDestroy(stream[i]));
		GPU_CHECKERROR(cudaFreeHost(inBuffer[i]));
		GPU_CHECKERROR(cudaFreeHost(outBuffer[i]));
		GPU_CHECKERROR(cudaFree(d_In[i]));
		GPU_CHECKERROR(cudaFree(d_Out[i]));
	}

	delete gridDim;
	delete blockDim;

	delete stream;
	delete currentInChunkSize;
	delete currentOutChunkSize;

	delete d_In;
	delete d_Out;

	delete inBuffer;

	fclose(inFile);
	fclose(outFile);

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

void GpuRSA::loadPublicKey(const char *publicKey)
{
	unsigned long long h_N[32], h_E[32];

	memset(h_N, 0, arrSize);
	memset(h_E, 0, arrSize);

	FILE *inFile = fopen(publicKey, "rb");
	if (inFile == NULL)
	{
		publicKeyReady = false;
	}

	memset(h_N + 16, 0, halfArrSize);
	if (fread(h_N, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		publicKeyReady = false;
	}

	memset(h_E + 16, 0, halfArrSize);
	if (fread(h_E, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		publicKeyReady = false;
	}

	fclose(inFile);

	GPU_CHECKERROR(cudaMemcpy(N, h_N, arrSize, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(E, h_E, arrSize, cudaMemcpyHostToDevice));

	publicKeyReady = true;
}

void GpuRSA::loadPrivateKey(const char *privateKey)
{
	unsigned long long h_N[32], h_D[32];

	memset(h_N, 0, arrSize);
	memset(h_D, 0, arrSize);

	FILE *inFile = fopen(privateKey, "rb");
	if (inFile == NULL)
	{
		privateKeyReady = false;
	}

	memset(h_N + 16, 0, halfArrSize);
	if (fread(h_N, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		privateKeyReady = false;
	}

	memset(h_D + 16, 0, halfArrSize);
	if (fread(h_D, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		privateKeyReady = false;
	}

	fclose(inFile);

	GPU_CHECKERROR(cudaMemcpy(N, h_N, arrSize, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaMemcpy(D, h_D, arrSize, cudaMemcpyHostToDevice));

	privateKeyReady = true;
}

__device__ void GpuRSA::add(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int tx = threadIdx.x;

	unsigned long long tmp = in1[tx] + in2[tx];
	bool carry = tmp < in1[tx];

	volatile unsigned long long *v_out = out + tx;
	*v_out = tmp;

	if (carry && tx < 31)
	{
		++out[tx + 1];

		for (int i = tx + 1; i < 31 && out[i] == 0; ++i)
		{
			++out[i + 1];
		}
	}

	__syncthreads();
}

__device__ void GpuRSA::sub(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int tx = threadIdx.x;

	unsigned long long tmp = in1[tx] - in2[tx];
	bool carry = tmp > in1[tx];

	volatile unsigned long long *v_out = out + tx;
	*v_out = tmp;

	if (carry && tx < 31)
	{
		--out[tx + 1];

		for (int i = tx + 1; i < 31 && out[i] == 0xFFFFFFFFFFFFFFFF; ++i)
		{
			--out[i + 1];
		}
	}

	__syncthreads();
}

__device__ void GpuRSA::mul(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long buffer[32], carry1[33], carry2[33], cur1[32],
			cur2[32];

	volatile __shared__ int inLen1;
	volatile __shared__ bool swapInput;

	volatile unsigned long long *v_buffer = buffer + tx;

	volatile unsigned long long *v_carry1 = carry1 + tx;
	volatile unsigned long long *v_carry2 = carry2 + tx;
	volatile unsigned long long *v_cur1 = cur1 + tx;
	volatile unsigned long long *v_cur2 = cur2 + tx;

	if (tx == 0)
	{
		inLen1 = length(in1);
		int inLen2 = length(in2);

		swapInput = inLen2 < inLen1;

		if (swapInput)
		{
			inLen1 = inLen2;
		}
	}

	__syncthreads();

	if (swapInput)
	{
		unsigned long long *tmp = in1;
		in1 = in2;
		in2 = tmp;
	}

	unsigned long long high1, high2, low1, low2, middle1, middle2;

	*v_buffer = 0;

	for (int i = 0; i < inLen1; ++i)
	{
		*v_cur1 = *v_cur2 = *v_carry1 = *v_carry2 = 0;

		high1 = in1[i] >> 32;
		low1 = in1[i] & 0xFFFFFFFF;

		if (i + tx < 32)
		{
			high2 = in2[tx] >> 32;
			low2 = in2[tx] & 0xFFFFFFFF;

			middle1 = low1 * high2;
			middle2 = low2 * high1;

			v_carry1[i + 1] = high1 * high2;
			v_carry2[i + 1] = (middle1 >> 32) + (middle2 >> 32)
					+ (((middle1 & 0xFFFFFFFF) + (middle2 & 0xFFFFFFFF)) >> 32);
			v_cur1[i] = low1 * low2;
			v_cur2[i] = (middle1 + middle2) << 32;
		}

		__syncthreads();

		add(buffer, cur1, buffer);
		add(buffer, cur2, buffer);

		add(buffer, carry1, buffer);
		add(buffer, carry2, buffer);
	}

	out[tx] = *v_buffer;
}

__device__ void GpuRSA::div(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out1, unsigned long long *out2)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long quotient[32], remainder[32], tmp[32];
	volatile __shared__ int inLen1, inLen2, cmpResult;

	volatile unsigned long long *v_quotient = quotient + tx;
	volatile unsigned long long *v_remainder = remainder + tx;
	volatile unsigned long long *v_tmp = tmp + tx;

	if (tx == 0)
	{
		inLen1 = length(in1);
		inLen2 = length(in2);
	}

	__syncthreads();

	int diff;
	bool firstEnough, firstBinary;
	unsigned long long l, r, m;

	*v_quotient = 0;
	*v_remainder = in1[tx];

	while (inLen1 >= inLen2)
	{
		diff = inLen1 - inLen2;
		firstEnough = quotient[diff] == 0
				&& remainder[inLen1 - 1] >= in2[inLen2 - 1];

		if (firstEnough)
		{
			l = 0;
			r = remainder[inLen1 - 1] / in2[inLen2 - 1];

			*v_tmp = 0;
			m = l + ((r - l + 1) >> 1);

			if (r == 0xFFFFFFFFFFFFFFFF)
			{
				m = 0x8000000000000000;
				firstBinary = true;
			}

			while (l <= r && (firstBinary || r - l != 0xFFFFFFFFFFFFFFFF))
			{
				if (tx == 0)
				{
					tmp[diff] = m;
				}

				__syncthreads();

				mul(tmp, in2, tmp);

				if (tx == 0)
				{
					cmpResult = cmp(remainder, tmp);
				}

				__syncthreads();

				if (cmpResult >= 0)
				{
					if (tx == 0)
					{
						quotient[diff] = m;
					}

					__syncthreads();

					l = m + 1;
				}
				else
				{
					r = m - 1;
				}

				*v_tmp = 0;
				m = l + ((r - l + 1) >> 1);

				firstBinary = false;
			}

			if (quotient[diff] > 0)
			{
				*v_tmp = 0;

				if (tx == 0)
				{
					tmp[diff] = quotient[diff];
				}

				__syncthreads();

				mul(tmp, in2, tmp);
				sub(remainder, tmp, remainder);
			}
			else
			{
				firstEnough = false;
			}
		}

		if (!firstEnough)
		{
			if (diff == 0)
			{
				break;
			}

			--diff;

			l = 0;
			r = 0xFFFFFFFFFFFFFFFF;

			*v_tmp = 0;

			m = 0x8000000000000000;
			firstBinary = true;

			while (l <= r && (firstBinary || r - l != 0xFFFFFFFFFFFFFFFF))
			{
				if (tx == 0)
				{
					tmp[diff] = m;
				}

				__syncthreads();

				mul(tmp, in2, tmp);

				if (tx == 0)
				{
					cmpResult = cmp(remainder, tmp);
				}

				__syncthreads();

				if (cmpResult >= 0)
				{
					if (tx == 0)
					{
						quotient[diff] = m;
					}

					__syncthreads();

					l = m + 1;
				}
				else
				{
					r = m - 1;
				}

				*v_tmp = 0;
				m = l + ((r - l + 1) >> 1);

				firstBinary = false;
			}

			*v_tmp = 0;

			if (tx == 0)
			{
				tmp[diff] = quotient[diff];
			}

			__syncthreads();

			mul(tmp, in2, tmp);
			sub(remainder, tmp, remainder);
		}

		if (tx == 0)
		{
			inLen1 = length(remainder);
		}

		__syncthreads();
	}

	if (out1 != NULL)
	{
		out1[tx] = *v_quotient;
	}

	if (out2 != NULL)
	{
		out2[tx] = *v_remainder;
	}
}

/**
 * Shift right
 */
__device__ void GpuRSA::shr(unsigned long long *in, unsigned long long *out,
		int offset)
{
	int tx = threadIdx.x;

	unsigned long long tmp = 0;

	int eleOffset = offset >> 6;
	int bitOffset = offset & 63;
	int invBitOffset = 64 - bitOffset;

	if (tx + eleOffset < 32)
	{
		tmp = in[tx + eleOffset] >> bitOffset;

		if (invBitOffset < 64 && tx + eleOffset < 31)
		{
			tmp |= in[tx + eleOffset + 1] << invBitOffset;
		}
	}

	__syncthreads();

	out[tx] = tmp;
}

/**
 * Random one unsigned long long
 */
__device__ unsigned long long GpuRSA::randomUll(curandState *states)
{
	states += threadIdx.x;

	unsigned long long high, low;

	high = curand(states);
	low = curand(states);

	return (high << 32) | low;
}

/**
 * Random big integer
 */
__device__ void GpuRSA::random(unsigned long long *in, int len,
		curandState *states)
{
	int tx = threadIdx.x;

	in[tx] = tx < len ? randomUll(states) : 0;

	__syncthreads();

	if (tx == len - 1)
	{
		while (in[tx] == 0)
		{
			in[tx] = randomUll(states);
		}
	}

	__syncthreads();
}

/**
 * Extended Euclidean method to find modular multiplicative inverse
 * Reference: http://www.di-mgt.com.au/euclidean.html
 */
__device__ void GpuRSA::extGcd(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long u1[32], u3[32], v1[32], v3[32], t1[32],
			t3[32], q[32];

	volatile __shared__ bool zeroResult;

	volatile unsigned long long *v_u1 = u1 + tx;
	volatile unsigned long long *v_u3 = u3 + tx;

	volatile unsigned long long *v_v1 = v1 + tx;
	volatile unsigned long long *v_v3 = v3 + tx;

	volatile unsigned long long *v_t1 = t1 + tx;
	volatile unsigned long long *v_t3 = t3 + tx;

	bool iter;

	*v_u1 = *v_v1 = 0;

	*v_u3 = in1[tx];
	*v_v3 = in2[tx];

	iter = 1;

	if (tx == 0)
	{
		u1[0] = 1;
		zeroResult = zero(v3);
	}

	__syncthreads();

	while (!zeroResult)
	{
		div(u3, v3, q, t3);
		mul(q, v1, q);
		add(u1, q, t1);

		*v_u1 = *v_v1;
		*v_v1 = *v_t1;

		*v_u3 = *v_v3;
		*v_v3 = *v_t3;

		iter = !iter;

		if (tx == 0)
		{
			zeroResult = zero(v3);
		}

		__syncthreads();
	}

	if (!iter)
	{
		sub(in2, u1, out);
	}
	else
	{
		out[tx] = *v_u1;
	}
}

/**
 * Add two 2048-bit integers and shift right by 1024 bits and preserve the bits higher than 1024
 */
__device__ void GpuRSA::addShift(unsigned long long *in1,
		unsigned long long *in2, unsigned long long *out)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long tmp1[32], tmp2[32];

	if (tx < 16)
	{
		tmp1[tx] = in1[tx];
		tmp2[tx] = in2[tx];
	}
	else
	{
		tmp1[tx] = tmp2[tx] = 0;
	}

	__syncthreads();

	add(tmp1, tmp2, tmp1);

	if (tx < 16)
	{
		tmp1[tx] = tmp1[tx + 16];
		tmp1[tx + 16] = 0;

		tmp2[tx] = in1[tx + 16];
	}

	__syncthreads();

	add(tmp1, tmp2, tmp1);

	if (tx < 16)
	{
		tmp2[tx] = in2[tx + 16];
	}

	__syncthreads();

	add(tmp1, tmp2, tmp1);
	out[tx] = tmp1[tx];
}

/**
 * Montgomery modular multiplication
 * Reference: http://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
 */
__device__ void GpuRSA::powMod(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *in3, unsigned long long *out,
		unsigned long long *invR, unsigned long long *invP)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long base[32], buffer[32], k[32];

	volatile __shared__ bool bit;
	volatile __shared__ int len, bitLen;

	volatile unsigned long long *v_buffer = buffer + tx;

	if (tx == 0)
	{
		bitLen = bitLength(in2);
	}

	__syncthreads();

	mul(in1, d_R, base);
	div(base, in3, NULL, base);

	*v_buffer = d_R[tx];
	div(buffer, in3, NULL, buffer);

	for (int i = 0; i < bitLen; ++i)
	{
		if (tx == 0)
		{
			bit = getBit(in2, i);
		}

		__syncthreads();

		if (bit)
		{
			mul(base, buffer, buffer);

			k[tx] = tx < 16 ? buffer[tx] : 0;

			__syncthreads();

			mul(k, invP, k);

			if (tx > 15)
			{
				k[tx] = 0;
			}

			__syncthreads();

			sub(d_R, k, k);

			mul(k, in3, k);
			addShift(buffer, k, buffer);

			if (tx == 0)
			{
				len = length(buffer);
			}

			__syncthreads();

			while (len > 16)
			{
				sub(buffer, in3, buffer);

				if (tx == 0)
				{
					len = length(buffer);
				}

				__syncthreads();
			}
		}

		mul(base, base, base);

		k[tx] = tx < 16 ? base[tx] : 0;

		__syncthreads();

		mul(k, invP, k);

		if (tx > 15)
		{
			k[tx] = 0;
		}

		__syncthreads();

		sub(d_R, k, k);

		mul(k, in3, k);
		addShift(base, k, base);

		if (tx == 0)
		{
			len = length(base);
		}

		__syncthreads();

		while (len > 16)
		{
			sub(base, in3, base);

			if (tx == 0)
			{
				len = length(base);
			}

			__syncthreads();
		}
	}

	mul(buffer, invR, buffer);
	div(buffer, in3, NULL, out);
}

/**
 * Generate a random prime, check it with small prime numbers
 */
__global__ void gpuRandomPrimeWithoutMillerRabin(unsigned long long *out,
		unsigned long long *sub1, bool *isPrime, GpuRSA *rsa)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long remainder[32], s_out[32], s_sub1[32], tmp[32];
	__shared__ curandState states[32];
	__shared__ bool zeroResult;

	states[tx] = rsa->states[tx];
	tmp[tx] = 0;

	rsa->random(s_out, 8, states);

	if (tx == 0)
	{
		s_out[0] |= 1;
		s_out[7] |= 0x8000000000000000;

		tmp[0] = 1;
	}

	__syncthreads();

	rsa->sub(s_out, tmp, s_sub1);

	if (tx == 0)
	{
		tmp[0] = DEFAULT_E;
	}

	__syncthreads();

	rsa->div(s_sub1, tmp, NULL, remainder);

	if (tx == 0)
	{
		zeroResult = rsa->zero(remainder);
	}

	__syncthreads();

	*isPrime = !zeroResult;

	if (!zeroResult)
	{
		for (int i = 0; i < SMALL_PRIMES_COUNT; ++i)
		{
			if (tx == 0)
			{
				tmp[0] = d_SMALL_PRIMES[i];
			}

			__syncthreads();

			rsa->div(s_out, tmp, NULL, remainder);

			if (tx == 0)
			{
				zeroResult = rsa->zero(remainder);
			}

			__syncthreads();

			if (zeroResult)
			{
				*isPrime = false;
				break;
			}
		}
	}

	out[tx] = s_out[tx];
	sub1[tx] = s_sub1[tx];

	rsa->states[tx] = states[tx];
}

__global__ void gpuPrint(unsigned long long *in, int length, bool newline,
		GpuRSA *rsa)
{
	rsa->print(in, length);

	if (newline)
	{
		printf("\n");
	}
}

__global__ void gpuEquals(unsigned long long *in1, unsigned long long *in2,
		bool *equals, GpuRSA *rsa)
{
	*equals = rsa->cmp(in1, in2) == 0;
}

/**
 * Init Miller-Rabin
 */
__global__ void gpuMillerRabinInit(unsigned long long *in,
		unsigned long long *sub1, unsigned long long *z,
		unsigned long long *invR, unsigned long long *invP, int *s,
		bool *passTest, bool *isPrime, GpuRSA *rsa)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int tid = bx * blockDim.x + tx;

	__shared__ unsigned long long a[32], m[32], s_z[32], tmp[32], s_invR[32],
			s_invP[32], s_in[32], s_sub1[32];

	__shared__ curandState states[32];
	__shared__ int randLen, len, cmpResult, s_s;

	states[tx] = rsa->states[tid];

	s_in[tx] = in[tx];
	s_sub1[tx] = m[tx] = sub1[tx];
	tmp[tx] = 0;

	rsa->extGcd(d_R, s_in, s_invR);
	rsa->extGcd(s_in, d_R, s_invP);

	if (tx == 0)
	{
		s_s = 0;

		while (!rsa->getBit(s_sub1, s_s))
		{
			++s_s;
		}
	}

	__syncthreads();

	rsa->shr(m, m, s_s);

	if (tx == 0)
	{
		tmp[0] = 2;
		len = rsa->length(s_in);
	}

	__syncthreads();

	do
	{
		if (tx == 0)
		{
			randLen = curand(states) % len + 1;
		}

		__syncthreads();

		rsa->random(a, randLen, states);
		rsa->add(a, tmp, a);

		if (tx == 0)
		{
			if (s_sub1[len - 1] < 0xFFFFFFFFFFFFFFFF)
			{
				a[len - 1] %= (s_sub1[len - 1] + 1);
			}

			cmpResult = rsa->cmp(s_sub1, a);
		}

		__syncthreads();
	}
	while (cmpResult <= 0);

	rsa->powMod(a, m, s_in, s_z, s_invR, s_invP);

	if (tx == 0)
	{
		tmp[0] = 1;
		cmpResult = rsa->cmp(s_z, tmp);

		passTest[bx] = cmpResult == 0;
	}

	__syncthreads();

	z[tid] = s_z[tx];

	if (bx == 0)
	{
		invR[tx] = s_invR[tx];
		invP[tx] = s_invP[tx];

		if (tx == 0)
		{
			*s = s_s;
		}
	}

	rsa->states[tid] = states[tx];
}

__global__ void gpuMillerRabinLoop(unsigned long long *in,
		unsigned long long *sub1, unsigned long long *z,
		unsigned long long *invR, unsigned long long *invP, bool last,
		bool *passTest, bool *isPrime, GpuRSA *rsa)
{
	int bx = blockIdx.x;

	if (passTest[bx])
	{
		return;
	}

	int tx = threadIdx.x;

	int tid = bx * blockDim.x + tx;

	__shared__ unsigned long long s_z[32], tmp[32], s_invR[32], s_invP[32],
			s_in[32], s_sub1[32];

	__shared__ int cmpResult;

	s_in[tx] = in[tx];
	s_sub1[tx] = sub1[tx];
	tmp[tx] = 0;

	s_z[tx] = z[tid];

	s_invR[tx] = invR[tx];
	s_invP[tx] = invP[tx];

	if (tx == 0)
	{
		tmp[0] = 2;
		cmpResult = rsa->cmp(s_z, s_sub1);
	}

	__syncthreads();

	if (cmpResult == 0)
	{
		if (tx == 0)
		{
			passTest[bx] = true;
		}

		__syncthreads();
	}
	else
	{
		if (last)
		{
			if (tx == 0)
			{
				*isPrime = 0;
			}

			__syncthreads();
		}
		else
		{
			rsa->powMod(s_z, tmp, s_in, s_z, s_invR, s_invP);
			z[tid] = s_z[tx];
		}
	}
}

/**
 * Generate random big prime numbers
 */
void GpuRSA::randomPrime(unsigned long long *out, unsigned long long *sub1,
		unsigned long long *z, unsigned long long *invR,
		unsigned long long *invP, int *d_s, bool *passTest, bool *d_isPrime)
{
	bool h_isPrime;
	int s;

	// The CPU function will be divided to several kernels to prevent timeout error for kernel launch

	do
	{
		// Generate random prime number and check it with small prime numbers

		do
		{
			gpuRandomPrimeWithoutMillerRabin<<<1, 32>>>(out, sub1, d_isPrime,
					d_this);
			GPU_CHECKERROR(cudaGetLastError());

			GPU_CHECKERROR(
					cudaMemcpy(&h_isPrime, d_isPrime, 1,
							cudaMemcpyDeviceToHost));
		}
		while (!h_isPrime);

		GPU_CHECKERROR(cudaMemset(passTest, 0, 32));

		// Init Miller-Rabin

		gpuMillerRabinInit<<<32, 32>>>(out, sub1, z, invR, invP, d_s, passTest,
				d_isPrime, d_this);

		GPU_CHECKERROR(
				cudaMemcpy(&h_isPrime, d_isPrime, 1, cudaMemcpyDeviceToHost));
		GPU_CHECKERROR(
				cudaMemcpy(&s, d_s, sizeof(int), cudaMemcpyDeviceToHost));

		// Perform Miller-Rabin loops

		for (int i = 0; i < s; ++i)
		{
			gpuMillerRabinLoop<<<32, 32>>>(out, sub1, z, invR, invP, i == s - 1,
					passTest, d_isPrime, d_this);
			GPU_CHECKERROR(cudaGetLastError());
		}

		GPU_CHECKERROR(
				cudaMemcpy(&h_isPrime, d_isPrime, 1, cudaMemcpyDeviceToHost));
	}
	while (!h_isPrime);
}

/**
 * Generate N, e, d for the key pair
 */
__global__ void gpuGenNED(unsigned long long *P, unsigned long long *Q,
		unsigned long long *P_1, unsigned long long *Q_1, GpuRSA *rsa)
{
	int tx = threadIdx.x;

	__shared__ unsigned long long s_P[32], s_Q[32], s_P_1[32], s_Q_1[32],
			s_N[32], s_E[32], s_D[32], r[32];

	s_P[tx] = P[tx];
	s_Q[tx] = Q[tx];

	s_P_1[tx] = P_1[tx];
	s_Q_1[tx] = Q_1[tx];

	s_E[tx] = rsa->E[tx] = 0;

	if (tx == 0)
	{
		s_E[tx] = rsa->E[tx] = DEFAULT_E;
	}

	__syncthreads();

	rsa->mul(s_P, s_Q, s_N);
	rsa->mul(s_P_1, s_Q_1, r);
	rsa->extGcd(s_E, r, s_D);

	rsa->N[tx] = s_N[tx];
	rsa->D[tx] = s_D[tx];
}

/**
 * Generate public and private key
 */
float GpuRSA::generateKeys(const char *publicKey, const char *privateKey)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	float gpuTime;
	cudaEvent_t gpuStart, gpuEnd;

	GPU_CHECKERROR(cudaEventCreate(&gpuStart));
	GPU_CHECKERROR(cudaEventCreate(&gpuEnd));

	GPU_CHECKERROR(cudaEventRecord(gpuStart, 0));

	unsigned long long *P, *Q, *P_1, *Q_1, *z, *invR, *invP;
	bool *d_isPrime, *passTest;
	int *d_s;

	GPU_CHECKERROR(cudaMalloc(&P, arrSize));
	GPU_CHECKERROR(cudaMalloc(&P_1, arrSize));

	GPU_CHECKERROR(cudaMalloc(&Q, arrSize));
	GPU_CHECKERROR(cudaMalloc(&Q_1, arrSize));

	GPU_CHECKERROR(cudaMalloc(&d_isPrime, 1));
	GPU_CHECKERROR(cudaMalloc(&passTest, 32));

	GPU_CHECKERROR(cudaMalloc(&d_s, sizeof(int)));
	GPU_CHECKERROR(cudaMalloc(&z, arrSize));

	GPU_CHECKERROR(cudaMalloc(&invR, arrSize));
	GPU_CHECKERROR(cudaMalloc(&invP, arrSize));

	// Generate big prime numbers P, Q

	randomPrime(P, P_1, z, invR, invP, d_s, passTest, d_isPrime);

	bool h_PeqQ, *d_PeqQ;

	GPU_CHECKERROR(cudaMalloc(&d_PeqQ, 1));

	do
	{
		randomPrime(Q, Q_1, z, invR, invP, d_s, passTest, d_isPrime);

		gpuEquals<<<1, 1>>>(P, Q, d_PeqQ, d_this);
		GPU_CHECKERROR(cudaGetLastError());

		GPU_CHECKERROR(cudaMemcpy(&h_PeqQ, d_PeqQ, 1, cudaMemcpyDeviceToHost));
	}
	while (h_PeqQ);

	GPU_CHECKERROR(cudaFree(d_PeqQ));

	// Compute N = PQ, e = 65537, ed % (P - 1)(Q - 1) = 1

	gpuGenNED<<<1, 32>>>(P, Q, P_1, Q_1, d_this);
	GPU_CHECKERROR(cudaGetLastError());

	unsigned long long h_N[32], h_E[32], h_D[32];

	GPU_CHECKERROR(cudaMemcpy(h_N, N, arrSize, cudaMemcpyDeviceToHost));
	GPU_CHECKERROR(cudaMemcpy(h_E, E, arrSize, cudaMemcpyDeviceToHost));
	GPU_CHECKERROR(cudaMemcpy(h_D, D, arrSize, cudaMemcpyDeviceToHost));

	GPU_CHECKERROR(cudaFree(d_isPrime));
	GPU_CHECKERROR(cudaFree(passTest));

	GPU_CHECKERROR(cudaFree(d_s));
	GPU_CHECKERROR(cudaFree(z));

	GPU_CHECKERROR(cudaFree(invR));
	GPU_CHECKERROR(cudaFree(invP));

	GPU_CHECKERROR(cudaFree(P));
	GPU_CHECKERROR(cudaFree(P_1));

	GPU_CHECKERROR(cudaFree(Q));
	GPU_CHECKERROR(cudaFree(Q_1));

	GPU_CHECKERROR(cudaDeviceSynchronize());

	// Output public and private key

	publicKeyReady = privateKeyReady = true;

	FILE *outFile = fopen(publicKey, "wb");
	if (outFile == NULL)
	{
		return OUT_FILE_ERROR;
	}

	if (fwrite(h_N, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}
	if (fwrite(h_E, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}

	fclose(outFile);

	outFile = fopen(privateKey, "wb");
	if (outFile == NULL)
	{
		return OUT_FILE_ERROR;
	}

	if (fwrite(h_N, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}

	if (fwrite(h_D, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}

	fclose(outFile);

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

}/* namespace RSA */
