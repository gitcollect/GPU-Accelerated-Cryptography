/*
 * GpuAES.cpp
 *
 *  Created on: 2015-4-13
 *      Author: Yuqing Guan
 */

#include "GpuAES.h"

namespace AES
{

__constant__ char d_SBOX[16][16], d_INV_SBOX[16][16];
__constant__ char d_RCON[16][4];

__constant__ char d_MUL02[256], d_MUL03[256], d_MUL09[256];
__constant__ char d_MUL0B[256], d_MUL0D[256], d_MUL0E[256];

GpuAES::GpuAES()
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

	maxThreadY = (int) ceil(prop.maxThreadsPerBlock / 16.0);
	maxBlockX = prop.maxGridSize[0];

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_SBOX, SBOX, sizeof(SBOX)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_INV_SBOX, INV_SBOX, sizeof(INV_SBOX)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_RCON, RCON, sizeof(RCON)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_MUL02, MUL02, sizeof(MUL02)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_MUL03, MUL03, sizeof(MUL03)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_MUL09, MUL09, sizeof(MUL09)));

	GPU_CHECKERROR(cudaMemcpyToSymbol(d_MUL0B, MUL0B, sizeof(MUL0B)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_MUL0D, MUL0D, sizeof(MUL0D)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_MUL0E, MUL0E, sizeof(MUL0E)));

	GPU_CHECKERROR(cudaMalloc(&w, 240 * 4));
}

GpuAES::~GpuAES()
{
	GPU_CHECKERROR(cudaFree(w));
}

__global__ void gpuSetKey(char *w, int Nk, int wLen)
{
	int tx = threadIdx.x;

	extern __shared__ char subkey[];
	volatile __shared__ char buffer[3][4];

	if (tx < Nk * 4)
	{
		subkey[tx] = w[tx];
	}

	__syncthreads();

	int row, col;

	if (tx < 4)
	{
		for (int i = Nk; i < wLen; ++i)
		{

			buffer[0][tx] = subkey[((i - 1) << 2) + tx];

			if (i % Nk == 0)
			{
				buffer[1][tx] = buffer[0][(tx + 1) & 3];

				row = (buffer[1][tx] >> 4) & 15;
				col = buffer[1][tx] & 15;
				buffer[2][tx] = d_SBOX[row][col];

				buffer[2][tx] ^= d_RCON[i / Nk][tx];
			}
			else if (Nk > 6 && i % Nk == 4)
			{
				row = (buffer[0][tx] >> 4) & 15;
				col = buffer[0][tx] & 15;
				buffer[2][tx] = d_SBOX[row][col];
			}
			else
			{
				buffer[2][tx] = buffer[0][tx];
			}

			subkey[(i << 2) + tx] = buffer[2][tx]
					^ subkey[((i - Nk) << 2) + tx];
		}
	}

	__syncthreads();

	if (tx >= Nk * 4)
	{
		w[tx] = subkey[tx];
	}
}

/**
 * Key expansion
 */
float GpuAES::setKey(const char *key)
{
	if (!deviceReady)
	{
		return DEVICE_ERROR;
	}

	int length = strlen(key);
	if (length != 16 && length != 24 && length != 32)
	{
		keyReady = false;
		return KEY_ERROR;
	}

	float gpuTime;
	cudaEvent_t gpuStart, gpuEnd;

	GPU_CHECKERROR(cudaEventCreate(&gpuStart));
	GPU_CHECKERROR(cudaEventCreate(&gpuEnd));

	GPU_CHECKERROR(cudaEventRecord(gpuStart, 0));

	Nk = length >> 2;
	Nr = Nk + 6;

	char *h_Key;
	GPU_CHECKERROR(cudaHostAlloc(&h_Key, length, cudaHostAllocDefault));
	memcpy(h_Key, key, length);

	int wLen = 4 * (Nr + 1);

	GPU_CHECKERROR(cudaMemcpy(w, h_Key, length, cudaMemcpyHostToDevice));
	GPU_CHECKERROR(cudaFreeHost(h_Key));

	gpuSetKey<<<1, wLen * 4, wLen * 4>>>(w, Nk, wLen);
	GPU_CHECKERROR(cudaGetLastError());

	GPU_CHECKERROR(cudaDeviceSynchronize());

	keyReady = true;

	GPU_CHECKERROR(cudaEventRecord(gpuEnd, 0));
	GPU_CHECKERROR(cudaEventSynchronize(gpuEnd));

	GPU_CHECKERROR(cudaEventElapsedTime(&gpuTime, gpuStart, gpuEnd));

	return gpuTime;
}

__global__ void gpuEncrypt(char *in, char *out, char *w, int Nr)
{
	char tmp;

	int times = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y
			+ threadIdx.y;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	extern __shared__ char allBytes[];
	volatile char *bytes = allBytes + ty * 16;
	volatile char *buffer = bytes + (tx & 12);

	bytes[tx] = in[tx + times * 16] ^ w[tx]; // Add round key

	int row, col;

	for (int i = 1; i < Nr; ++i)
	{
		row = (bytes[tx] >> 4) & 15;
		col = bytes[tx] & 15;
		bytes[tx] = d_SBOX[row][col]; // Sub bytes

		col = tx & 3;
		bytes[tx] = bytes[((col << 2) + tx) & 15]; // Shift rows

		tmp = d_MUL02[(unsigned char) buffer[col]]
				^ d_MUL03[(unsigned char) buffer[(col + 1) & 3]]
				^ buffer[(col + 2) & 3] ^ buffer[(col + 3) & 3]; // Mix columns

		bytes[tx] = tmp ^ w[(i << 4) + tx]; // Add round key
	}

	row = (bytes[tx] >> 4) & 15;
	col = bytes[tx] & 15;
	bytes[tx] = d_SBOX[row][col]; // Sub bytes

	col = tx & 3;
	tmp = bytes[((col << 2) + tx) & 15]; // Shift rows
	out[tx + times * 16] = tmp ^ w[(Nr << 4) + tx]; // Add round key
}

__global__ void gpuDecrypt(char *in, char *out, char *w, int Nr)
{
	char tmp;

	int times = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y
			+ threadIdx.y;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	extern __shared__ char allBytes[];
	volatile char *bytes = allBytes + ty * 16;
	volatile char *buffer = bytes + (tx & 12);

	bytes[tx] = in[tx + times * 16] ^ w[(Nr << 4) + tx]; // Add round key

	int row, col;

	for (int i = Nr - 1; i > 0; --i)
	{
		col = tx & 3;
		tmp = bytes[(16 - (col << 2) + tx) & 15]; // Invert shift rows

		row = (tmp >> 4) & 15;
		col = tmp & 15;
		tmp = d_INV_SBOX[row][col]; // Invert sub bytes

		bytes[tx] = tmp ^ w[(i << 4) + tx]; // Add round key

		col = tx & 3;
		bytes[tx] = d_MUL0E[(unsigned char) buffer[col]]
				^ d_MUL0B[(unsigned char) buffer[(col + 1) & 3]]
				^ d_MUL0D[(unsigned char) buffer[(col + 2) & 3]]
				^ d_MUL09[(unsigned char) buffer[(col + 3) & 3]]; // Invert mix columns
	}

	col = tx & 3;
	tmp = bytes[(16 - (col << 2) + tx) & 15]; // Invert shift rows

	row = (tmp >> 4) & 15;
	col = tmp & 15; // Invert sub bytes

	out[tx + times * 16] = d_INV_SBOX[row][col] ^ w[tx]; // Add round key
}

float GpuAES::encryptFile(const char *inName, const char *outName)
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
				cudaHostAlloc(buffer + i, chunkSize + 16,
						cudaHostAllocDefault));
		GPU_CHECKERROR(cudaMalloc(d_In + i, chunkSize));
		GPU_CHECKERROR(cudaMalloc(d_Out + i, chunkSize + 16));
	}

	int times = chunkSize / 16;

	int threadY = min(times, maxThreadY);
	int blocks = (int) ceil(times / (double) maxThreadY);

	int blockX = min(blocks, maxBlockX);
	int blockY = (int) ceil(blocks / (double) maxBlockX);

	dim3 defaultGridDim(blockX, blockY, 1);
	dim3 defaultBlockDim(16, threadY, 1);

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

			times = currentChunkSize[i] / 16 + (pos == size);
			int newLength = times * 16;
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
				blockDim[i] = dim3(16, threadY, 1);
			}
			else
			{
				gridDim[i] = defaultGridDim;
				blockDim[i] = defaultBlockDim;
			}
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			gpuEncrypt<<<gridDim[i], blockDim[i], blockDim[i].y * 16, stream[i]>>>(
					d_In[i], d_Out[i], w, Nr);
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

float GpuAES::decryptFile(const char *inName, const char *outName)
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

	int times = chunkSize / 16;

	int threadY = min(times, maxThreadY);
	int blocks = (int) ceil(times / (double) maxThreadY);

	int blockX = min(blocks, maxBlockX);
	int blockY = (int) ceil(blocks / (double) maxBlockX);

	dim3 defaultGridDim(blockX, blockY, 1);
	dim3 defaultBlockDim(16, threadY, 1);

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

			times = currentChunkSize[i] / 16;

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
				blockDim[i] = dim3(16, threadY, 1);
			}
			else
			{
				gridDim[i] = defaultGridDim;
				blockDim[i] = defaultBlockDim;
			}
		}

		for (int i = 0; i < runningStreams; ++i)
		{
			gpuDecrypt<<<gridDim[i], blockDim[i], blockDim[i].y * 16, stream[i]>>>(
					d_In[i], d_Out[i], w, Nr);
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
