/*
 * CpuDES.cpp
 *
 *  Created on: 2015-4-11
 *      Author: Yuqing Guan
 */

#include "CpuDES.h"
using namespace std;

namespace DES
{

/**
 * Convert a byte array to a bit array
 */
void CpuDES::bytes2Bits(const char *in, bool *out, int length)
{
	for (int i = 0; i < length; ++i)
	{
		out[i] = (in[i / 8] >> (7 - i % 8)) & 1;
	}
}

/**
 * Convert a bit array to a byte array
 */
void CpuDES::bits2Bytes(bool *in, char *out, int length)
{
	memset(out, 0, length / 8);

	for (int i = 0; i < length; ++i)
	{
		out[i / 8] |= (in[i] << (7 - i % 8));
	}
}

/**
 * Convert a hex string to a bit array
 */
void CpuDES::hex2Bits(char *in, bool *out, int length)
{
	for (int i = 0; i < length; ++i)
	{
		out[i] = ((in[i / 4] - (in[i / 4] > '9' ? '7' : '0')) >> (3 - i % 4)) & 1;
	}
}

/**
 * Convert a bit array to a hex string
 */
void CpuDES::bits2Hex(bool *in, char *out, int length)
{
	memset(out, 0, length / 4 + 1);

	for (int i = 0; i < length / 4; ++i)
	{
        out[i] = in[i * 4 + 3] + (in[i * 4 + 2] << 1) + (in[i * 4 + 1] << 2) + (in[i * 4] << 3);
        out[i] += out[i] > 9 ? '7' : '0';
	}
}

/**
 * Permute a table
 */
void CpuDES::permute(bool *in, bool *out, const char *table, int length)
{
	bool *buffer = new bool[length];

	for (int i = 0; i < length; ++i)
	{
		buffer[i] = in[table[i] - 1];
	}
	memcpy(out, buffer, length);

	delete buffer;
}

/**
 * Rotate left an array
 */
void CpuDES::rol(bool *array, int length, int shift)
{
	bool *buffer = new bool[length];

	memcpy(buffer + length - shift, array, shift);
	memcpy(buffer, array + shift, length - shift);
	memcpy(array, buffer, length);

	delete buffer;
}

/**
 * Xor two arrays
 */
void CpuDES::xorBlock(bool *in1, bool *in2, bool *out, int length)
{
	for (int i = 0; i < length; ++i)
	{
		out[i] = in1[i] ^ in2[i];
	}
}

/**
 * Feistel Functoin
 */
void CpuDES::feistel(bool *array, int round)
{
	bool *buffer = new bool[48];

	permute(array, buffer, E, 48);

	xorBlock(subkey[round], buffer, buffer, 48);

	int x, y;
	x = y = 0;

	bool *in, *out;

	in = buffer;
	out = array;

	for (int i = 0; i < 8; ++i)
	{
		y = (in[0] << 1) + in[5];
		x = (in[1] << 3) + (in[2] << 2) + (in[3] << 1) + in[4];

		for (int j = 0; j < 4; ++j)
		{
			out[j] = (SBOX[i][y][x] >> (3 - j)) & 1;
		}

        out += 4;
        in += 6;
	}

	permute(array, array, P, 32);

	delete buffer;
}

/**
 * Encrypt a block
 */
void CpuDES::encryptBlock(char *in, char *out)
{
	bool *bits = new bool[64];
	bool *buffer = new bool[32];

	bytes2Bits(in, bits, 64);

	permute(bits, bits, IP, 64);

	bool *low = bits;
	bool *high = bits + 32;

	for (int i = 0; i < 16; ++i)
	{
		memcpy(buffer, high, 32);
		feistel(high, i);
		xorBlock(low, high, high, 32);
		memcpy(low, buffer, 32);
	}

	rol(bits, 64, 32);

	permute(bits, bits, FP, 64);
	bits2Bytes(bits, out, 64);

	delete bits;
	delete buffer;
}

/**
 * Encrypt a block
 */
void CpuDES::decryptBlock(char *in, char *out)
{
	bool *bits = new bool[64];
	bool *buffer = new bool[32];

	bytes2Bits(in, bits, 64);
	permute(bits, bits, IP, 64);

	bool *low = bits;
	bool *high = bits + 32;

	rol(bits, 64, 32);

	for (int i = 15; i >= 0; --i)
	{
		memcpy(buffer, low, 32);
		feistel(low, i);
		xorBlock(low, high, low, 32);
		memcpy(high, buffer, 32);
	}

	permute(bits, bits, FP, 64);
	bits2Bytes(bits, out, 64);

	delete bits;
	delete buffer;
}

/**
 * Set symmetric key
 */
float CpuDES::setKey(const char *key)
{
	int length = strlen(key);
	if (length != 8)
	{
		keyReady = false;
		return KEY_ERROR;
	}

	clock_t start = clock();

	bool *bits = new bool[64];

	bytes2Bits(key, bits, 64);
	permute(bits, bits, PC1, 56);

	bool *low = bits;
	bool *high = bits + 28;

	for (int i = 0; i < 16; ++i)
	{
		rol(low, 28, SHIFT[i]);
		rol(high, 28, SHIFT[i]);

		permute(bits, subkey[i], PC2, 48);
	}

	delete bits;

	keyReady = true;

	return (clock() - start) / CLOCK_PER_MILL;
}

/**
 * Encrypt file
 */
float CpuDES::encryptFile(const char *inName, const char *outName)
{
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

	clock_t start = clock();

	char **buffer = new char*[maxStoredChunks];

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		buffer[i] = new char[chunkSize + 8];
	}

	fseek(inFile, 0, SEEK_SET);

	size_t pos = 0, currentChunkSize;

	while (pos < size)
	{
		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentChunkSize = chunkSize;
			if (pos + currentChunkSize > size)
			{
				currentChunkSize = size - pos;
			}

			if (fread(buffer[i], 1, currentChunkSize, inFile) != currentChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentChunkSize;

			int numBlocks = currentChunkSize / 8 + (pos == size);
			int newLength = numBlocks * 8;
			int padding = newLength - currentChunkSize;

			if (padding > 0)
			{
				memset(buffer[i] + currentChunkSize, padding, padding);
				currentChunkSize = newLength;
			}

			int offset = 0;

			for (int j = 0; j < numBlocks; ++j)
			{
				encryptBlock(buffer[i] + offset, buffer[i] + offset);
				offset += 8;
			}

			if (fwrite(buffer[i], 1, currentChunkSize, outFile) != currentChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return OUT_FILE_ERROR;
			}
		}
	}

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		delete buffer[i];
	}

	delete buffer;

	fclose(inFile);
	fclose(outFile);

	return (clock() - start) / CLOCK_PER_MILL;
}

/**
 * Decrypt file
 */
float CpuDES::decryptFile(const char *inName, const char *outName)
{
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

	clock_t start = clock();

	char **buffer = new char*[maxStoredChunks];

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		buffer[i] = new char[chunkSize];
	}

	fseek(inFile, 0, SEEK_SET);

	size_t pos = 0, currentChunkSize;

	while (pos < size)
	{
		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentChunkSize = chunkSize;
			if (pos + currentChunkSize > size)
			{
				currentChunkSize = size - pos;
			}

			if (fread(buffer[i], 1, currentChunkSize, inFile) != currentChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentChunkSize;

			int numBlocks = currentChunkSize / 8;
			int offset = 0;

			for (int j = 0; j < numBlocks; ++j)
			{
				decryptBlock(buffer[i] + offset, buffer[i] + offset);
				offset += 8;
			}

			if (pos == size)
			{
				currentChunkSize -= buffer[i][currentChunkSize - 1];
			}

			if (fwrite(buffer[i], 1, currentChunkSize, outFile) != currentChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return OUT_FILE_ERROR;
			}
		}
	}

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		delete buffer[i];
	}

	delete buffer;

	fclose(inFile);
	fclose(outFile);

	return (clock() - start) / CLOCK_PER_MILL;
}

}
