/*
 * GpuDES.cpp
 *
 *  Created on: 2015-4-13
 *      Author: Yuqing Guan
 */

#include "GpuDES.h"

namespace DES
{

// Constant memory

__constant__ char d_IP[64], d_FP[64];
__constant__ char d_PC1[56], d_PC2[48];
__constant__ char d_SHIFT[16];
__constant__ char d_E[48], d_P[32];
__constant__ char d_SBOX[8][4][16];

/**
 * Check GPU device, allocate constant and global memory
 */
GpuDES::GpuDES()
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

	maxThreadY = (int) ceil(prop.maxThreadsPerBlock / 64.0);
	maxBlockX = prop.maxGridSize[0];

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_IP, IP, sizeof(IP)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_FP, FP, sizeof(FP)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_PC1, PC1, sizeof(PC1)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_PC2, PC2, sizeof(PC2)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_SHIFT, SHIFT, sizeof(SHIFT)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_E, E, sizeof(E)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_P, P, sizeof(P)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_SBOX, SBOX, sizeof(SBOX)));

	GPU_CHECKERROR(cudaMalloc(&subkey, 16 * 48));
}

/**
 * Release global memory
 */
GpuDES::~GpuDES()
{
	GPU_CHECKERROR(cudaFree(subkey));
}

/**
 * Set symmetric key on GPU
 */
__global__ void gpuSetKey(char *key, bool *subkey)
{
	bool tmp;

	int tx = threadIdx.x;
	extern __shared__ bool bits[];

	bits[tx] = (key[tx / 8] >> (7 - tx % 8)) & 1;

	__syncthreads();

	if (tx < 56)
	{
		tmp = bits[d_PC1[tx] - 1];
	}

	__syncthreads();

	if (tx < 56)
	{
		bits[tx] = tmp;
	}

	__syncthreads();

	for (int i = 0; i < 16; ++i)
	{
		int halfIdx = tx % 28;
		int newHalfIdx = (halfIdx + d_SHIFT[i]) % 28;

		int newIdx = 28 * (tx / 28) + newHalfIdx;

		if (tx < 56)
		{
			tmp = bits[newIdx];
		}

		__syncthreads();

		if (tx < 56)
		{
			bits[tx] = tmp;
		}

		__syncthreads();

		if (tx < 48)
		{
			subkey[i * 48 + tx] = bits[d_PC2[tx] - 1];
		}

		__syncthreads();
	}
}

/**
 * Set symmetric key
 */
float GpuDES::setKey(const char *key)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	int length = strlen(key);
	if (length != 8)
	{
		keyReady = false;
		return KEY_ERROR;
	}

	float gpuTime;
	cudaEvent_t gpuStart, gpuEnd;

	GPU_CHECKERROR(cudaEventCreate(&gpuStart));
	GPU_CHECKERROR(cudaEventCreate(&gpuEnd));

	GPU_CHECKERROR(cudaEventRecord(gpuStart, 0));

	char *h_Key, *d_Key;
	GPU_CHECKERROR(cudaHostAlloc(&h_Key, length, cudaHostAllocDefault));
	memcpy(h_Key, key, length);

	GPU_CHECKERROR(cudaMalloc(&d_Key, length));
	GPU_CHECKERROR(cudaMemcpy(d_Key, h_Key, length, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaFreeHost(h_Key));

	gpuSetKey<<<1, 64, 64>>>(d_Key, subkey);
	GPU_CHECKERROR(cudaGetLastError());

	GPU_CHECKERROR(cudaFree(d_Key));
	GPU_CHECKERROR(cudaDeviceSynchronize());

	keyReady = true;

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

/**
 * Encrypt on GPU
 */
__global__ void gpuEncrypt(char *in, char *out, bool *subkey)
{
	bool tmp;

	int times = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y
			+ threadIdx.y;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	extern __shared__ bool allBits[];
	volatile bool *bits = allBits + ty * 112;

	bits[tx] = (in[tx / 8 + times * 8] >> (7 - tx % 8)) & 1;

	__syncthreads();

	tmp = bits[d_IP[tx] - 1];

	__syncthreads();

	bits[tx] = tmp;

	__syncthreads();

	volatile bool *high = bits + 32;
	volatile bool *buffer = bits + 64;
	volatile bool *fIn = buffer + (tx >> 2) * 6;

	int x, y;

	for (int i = 0; i < 16; ++i)
	{
		if (tx < 32)
		{
			tmp = high[tx];
		}

		if (tx < 48)
		{
			buffer[tx] = high[d_E[tx] - 1] ^ subkey[i * 48 + tx];
		}

		__syncthreads();

		if (tx < 32)
		{
			y = (fIn[0] << 1) + fIn[5];
			x = (fIn[1] << 3) + (fIn[2] << 2) + (fIn[3] << 1) + fIn[4];

			high[tx] = (d_SBOX[tx >> 2][y][x] >> (3 - (tx & 3))) & 1;

			high[tx] = high[d_P[tx] - 1] ^ bits[tx];
			bits[tx] = tmp;
		}

		__syncthreads();
	}

	tmp = bits[(d_FP[tx] - 1) ^ 32];

	__syncthreads();

	bits[tx] = tmp;

	__syncthreads();

	if (tx < 8)
	{
		ty = tx << 3;
		out[tx + times * 8] = (bits[ty] << 7) + (bits[ty + 1] << 6)
				+ (bits[ty + 2] << 5) + (bits[ty + 3] << 4)
				+ (bits[ty + 4] << 3) + (bits[ty + 5] << 2)
				+ (bits[ty + 6] << 1) + bits[ty + 7];
	}
}

/**
 * Decrypt on GPU
 */
__global__ void gpuDecrypt(char *in, char *out, bool *subkey)
{
	bool tmp;

	int times = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y
			+ threadIdx.y;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	extern __shared__ bool allBits[];
	volatile bool *bits = allBits + ty * 112;

	bits[tx] = (in[tx / 8 + times * 8] >> (7 - tx % 8)) & 1;

	__syncthreads();

	tmp = bits[d_IP[tx] - 1];

	__syncthreads();

	bits[tx ^ 32] = tmp;

	__syncthreads();

	volatile bool *high = bits + 32;
	volatile bool *buffer = bits + 64;
	volatile bool *fIn = buffer + (tx >> 2) * 6;

	int x, y;

	for (int i = 15; i >= 0; --i)
	{
		if (tx < 32)
		{
			tmp = bits[tx];
		}

		if (tx < 48)
		{
			buffer[tx] = bits[d_E[tx] - 1] ^ subkey[i * 48 + tx];
		}

		__syncthreads();

		if (tx < 32)
		{
			y = (fIn[0] << 1) + fIn[5];
			x = (fIn[1] << 3) + (fIn[2] << 2) + (fIn[3] << 1) + fIn[4];

			bits[tx] = (d_SBOX[tx >> 2][y][x] >> (3 - (tx & 3))) & 1;

			bits[tx] = bits[d_P[tx] - 1] ^ high[tx];
			high[tx] = tmp;
		}

		__syncthreads();
	}

	tmp = bits[d_FP[tx] - 1];

	__syncthreads();

	bits[tx] = tmp;

	__syncthreads();

	if (tx < 8)
	{
		ty = tx << 3;
		out[tx + times * 8] = (bits[ty] << 7) + (bits[ty + 1] << 6)
				+ (bits[ty + 2] << 5) + (bits[ty + 3] << 4)
				+ (bits[ty + 4] << 3) + (bits[ty + 5] << 2)
				+ (bits[ty + 6] << 1) + bits[ty + 7];
	}
}

/**
 * Encrypt file
 */
float GpuDES::encryptFile(const char *inName, const char *outName)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	if (!keyReady)
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

	int *currentChunkSize = new int[maxStoredChunks];
	cudaStream_t *stream = new cudaStream_t[maxStoredChunks];

	char **buffer = new char*[maxStoredChunks];

	char **d_In = new char*[maxStoredChunks];
	char **d_Out = new char*[maxStoredChunks];

	dim3 *gridDim = new dim3[maxStoredChunks];
	dim3 *blockDim = new dim3[maxStoredChunks];

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		GPU_CHECKERROR(cudaStreamCreate(stream + i));
		GPU_CHECKERROR(
				cudaHostAlloc(buffer + i, chunkSize + 8, cudaHostAllocDefault));
		GPU_CHECKERROR(cudaMalloc(d_In + i, chunkSize));
		GPU_CHECKERROR(cudaMalloc(d_Out + i, chunkSize + 8));
	}

	int times = chunkSize / 8;

	int threadY = min(times, maxThreadY);
	int blocks = (int) ceil(times / (double) maxThreadY);

	int blockX = min(blocks, maxBlockX);
	int blockY = (int) ceil(blocks / (double) maxBlockX);

	dim3 defaultGridDim(blockX, blockY, 1);
	dim3 defaultBlockDim(64, threadY, 1);

	while (pos < size)
	{
		runningStreams = 0;

		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentChunkSize[i] = chunkSize;
			if (pos + currentChunkSize[i] > size)
			{
				currentChunkSize[i] = size - pos;
			}

			if (fread(buffer[i], 1, currentChunkSize[i], inFile)
					!= currentChunkSize[i])
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentChunkSize[i];
			++runningStreams;

			times = currentChunkSize[i] / 8 + (pos == size);
			int newLength = times * 8;
			int padding = newLength - currentChunkSize[i];

			GPU_CHECKERROR(
					cudaMemcpyAsync(d_In[i], buffer[i], currentChunkSize[i],
							cudaMemcpyHostToDevice, stream[i]));

			if (pos == size)
			{
				GPU_CHECKERROR(
						cudaMemsetAsync(d_In[i] + currentChunkSize[i], padding,
								padding, stream[i]));
				currentChunkSize[i] += padding;

				threadY = min(times, maxThreadY);
				blocks = (int) ceil(times / (double) maxThreadY);

				blockX = min(blocks, maxBlockX);
				blockY = (int) ceil(blocks / (double) maxBlockX);

				gridDim[i] = dim3(blockX, blockY, 1);
				blockDim[i] = dim3(64, threadY, 1);
			}
			else
			{
				gridDim[i] = defaultGridDim;
				blockDim[i] = defaultBlockDim;
			}
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			gpuEncrypt<<<gridDim[i], blockDim[i], blockDim[i].y * 112, stream[i]>>>(
					d_In[i], d_Out[i], subkey);
			GPU_CHECKERROR(cudaGetLastError());
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(
					cudaMemcpyAsync(buffer[i], d_Out[i], currentChunkSize[i],
							cudaMemcpyDeviceToHost, stream[i]));
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(cudaStreamSynchronize(stream[i]));

			if (fwrite(buffer[i], 1, currentChunkSize[i], outFile)
					!= currentChunkSize[i])
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
		GPU_CHECKERROR(cudaFreeHost(buffer[i]));
		GPU_CHECKERROR(cudaFree(d_In[i]));
		GPU_CHECKERROR(cudaFree(d_Out[i]));
	}

	delete gridDim;
	delete blockDim;

	delete stream;
	delete currentChunkSize;

	delete d_In;
	delete d_Out;

	delete buffer;

	fclose(inFile);
	fclose(outFile);

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

/**
 * Decrypt file
 */
float GpuDES::decryptFile(const char *inName, const char *outName)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	if (!keyReady)
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

	if ((size & 7) != 0)
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

	int *currentChunkSize = new int[maxStoredChunks];
	cudaStream_t *stream = new cudaStream_t[maxStoredChunks];

	char **buffer = new char*[maxStoredChunks];

	char **d_In = new char*[maxStoredChunks];
	char **d_Out = new char*[maxStoredChunks];

	dim3 *gridDim = new dim3[maxStoredChunks];
	dim3 *blockDim = new dim3[maxStoredChunks];

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		GPU_CHECKERROR(cudaStreamCreate(stream + i));
		GPU_CHECKERROR(
				cudaHostAlloc(buffer + i, chunkSize, cudaHostAllocDefault));
		GPU_CHECKERROR(cudaMalloc(d_In + i, chunkSize));
		GPU_CHECKERROR(cudaMalloc(d_Out + i, chunkSize));
	}

	int times = chunkSize / 8;

	int threadY = min(times, maxThreadY);
	int blocks = (int) ceil(times / (double) maxThreadY);

	int blockX = min(blocks, maxBlockX);
	int blockY = (int) ceil(blocks / (double) maxBlockX);

	dim3 defaultGridDim(blockX, blockY, 1);
	dim3 defaultBlockDim(64, threadY, 1);

	while (pos < size)
	{
		runningStreams = 0;

		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentChunkSize[i] = chunkSize;
			if (pos + currentChunkSize[i] > size)
			{
				currentChunkSize[i] = size - pos;
			}

			if (fread(buffer[i], 1, currentChunkSize[i], inFile)
					!= currentChunkSize[i])
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentChunkSize[i];
			++runningStreams;

			times = currentChunkSize[i] / 8;

			GPU_CHECKERROR(
					cudaMemcpyAsync(d_In[i], buffer[i], currentChunkSize[i],
							cudaMemcpyHostToDevice, stream[i]));

			if (pos == size)
			{
				threadY = min(times, maxThreadY);
				blocks = (int) ceil(times / (double) maxThreadY);

				blockX = min(blocks, maxBlockX);
				blockY = (int) ceil(blocks / (double) maxBlockX);

				gridDim[i] = dim3(blockX, blockY, 1);
				blockDim[i] = dim3(64, threadY, 1);
			}
			else
			{
				gridDim[i] = defaultGridDim;
				blockDim[i] = defaultBlockDim;
			}
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			gpuDecrypt<<<gridDim[i], blockDim[i], blockDim[i].y * 112, stream[i]>>>(
					d_In[i], d_Out[i], subkey);
			GPU_CHECKERROR(cudaGetLastError());
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(
					cudaMemcpyAsync(buffer[i], d_Out[i], currentChunkSize[i],
							cudaMemcpyDeviceToHost, stream[i]));
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			GPU_CHECKERROR(cudaStreamSynchronize(stream[i]));

			if (pos == size && i == runningStreams - 1)
			{
				currentChunkSize[i] -= buffer[i][currentChunkSize[i] - 1];
			}

			if (fwrite(buffer[i], 1, currentChunkSize[i], outFile)
					!= currentChunkSize[i])
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
		GPU_CHECKERROR(cudaFreeHost(buffer[i]));
		GPU_CHECKERROR(cudaFree(d_In[i]));
		GPU_CHECKERROR(cudaFree(d_Out[i]));
	}

	delete gridDim;
	delete blockDim;

	delete stream;
	delete currentChunkSize;

	delete d_In;
	delete d_Out;

	delete buffer;

	fclose(inFile);
	fclose(outFile);

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

}

