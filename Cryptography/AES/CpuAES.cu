/*
 * CpuAES.cpp
 *
 *  Created on: 2015-4-12
 *      Author: Yuqing Guan
 */

#include "CpuAES.h"

namespace AES
{

/**
 * Convert a hex string to a byte array
 */
void CpuAES::hex2Bytes(char *in, char *out, int length)
{
	memset(out, 0, length + 1);

	for (int i = 0; i < length; ++i)
	{
		out[i] = ((in[i * 2] - (in[i * 2] > '9' ? '7' : '0')) << 4)
				+ (in[i * 2 + 1] - (in[i * 2 + 1] > '9' ? '7' : '0'));
	}
}

/**
 * Convert a bit array to a byte string
 */
void CpuAES::bytes2Hex(char *in, char *out, int length)
{
	memset(out, 0, length * 2 + 1);

	for (int i = 0; i < length; ++i)
	{
        out[i * 2] = (in[i] >> 4) & 15;
        out[i * 2 + 1] = in[i] & 15;

        out[i * 2] += out[i * 2] > 9 ? '7' : '0';
        out[i * 2 + 1] += out[i * 2 + 1] > 9 ? '7' : '0';
	}
}

/**
 * Rotate left an array
 */
void CpuAES::rol(char *array, int length, int shift)
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
void CpuAES::xorBlock(char *in1, char *in2, char *out, int length)
{
	for (int i = 0; i < length; ++i)
	{
		out[i] = in1[i] ^ in2[i];
	}
}

void CpuAES::subWord(char *bytes)
{
	for (int i = 0; i < 4; ++i)
	{
		int row = (bytes[i] >> 4) & 15;
		int col = bytes[i] & 15;
		bytes[i] = SBOX[row][col];
	}
}

/**
 * Key expansion
 */
float CpuAES::setKey(const char *key)
{
	int length = strlen(key);
	if (length != 16 && length != 24 && length != 32)
	{
		keyReady = false;
		return KEY_ERROR;
	}

	clock_t start = clock();

	Nk = length >> 2;
	Nr = Nk + 6;

	int i;
	for (i = 0; i < Nk; ++i)
	{
		memcpy(w[i], key + i * 4, 4);
	}

	int wLen = 4 * (Nr + 1);
	char buffer[4];

	for (; i < wLen; i++)
	{
		memcpy(buffer, w[i - 1], 4);

		if (i % Nk == 0)
		{
			rol(buffer, 4, 1);
			subWord(buffer);

			buffer[0] ^= RCON[i / Nk][0];
		}
		else if (Nk > 6 && i % Nk == 4)
		{
			subWord(buffer);
		}

		xorBlock(w[i - Nk], buffer, w[i], 4);
	}

	keyReady = true;

	return (clock() - start) / CLOCK_PER_MILL;
}

void CpuAES::addRoundKey(char *bytes, int round)
{
	for (int i = 0; i < 4; ++i)
	{
		xorBlock(bytes + i * 4, w[round * 4 + i], bytes + i * 4, 4);
	}
}

void CpuAES::subBytes(char *bytes)
{
	for (int i = 0; i < 16; ++i)
	{
		int row = (bytes[i] >> 4) & 15;
		int col = bytes[i] & 15;
		bytes[i] = SBOX[row][col];
	}
}

void CpuAES::invSubBytes(char *bytes)
{
	for (int i = 0; i < 16; ++i)
	{
		int row = (bytes[i] >> 4) & 15;
		int col = bytes[i] & 15;
		bytes[i] = INV_SBOX[row][col];
	}
}

void CpuAES::shiftRows(char *bytes)
{
	char buffer[4];
	for (int i = 1; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			buffer[j] = bytes[((j + i) % 4) * 4 + i];
		}

		for (int j = 0; j < 4; ++j)
		{
			bytes[j * 4 + i] = buffer[j];
		}
	}
}

void CpuAES::invShiftRows(char *bytes)
{
	char buffer[4];
	for (int i = 1; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			buffer[j] = bytes[((j + 4 - i) % 4) * 4 + i];
		}

		for (int j = 0; j < 4; ++j)
		{
			bytes[j * 4 + i] = buffer[j];
		}
	}
}

void CpuAES::mixColumns(char *bytes)
{
	char buffer[4];

	for (int i = 0; i < 4; ++i)
	{
		memcpy(buffer, bytes + i * 4, 4);

		bytes[i * 4] = MUL02[(unsigned char) buffer[0]] ^ MUL03[(unsigned char) buffer[1]] ^
				buffer[2] ^ buffer[3];
		bytes[i * 4 + 1] = buffer[0] ^ MUL02[(unsigned char) buffer[1]] ^
				MUL03[(unsigned char) buffer[2]] ^ buffer[3];
		bytes[i * 4 + 2] = buffer[0] ^ buffer[1] ^
				MUL02[(unsigned char) buffer[2]] ^ MUL03[(unsigned char) buffer[3]];
		bytes[i * 4 + 3] = MUL03[(unsigned char) buffer[0]] ^ buffer[1] ^
				buffer[2] ^ MUL02[(unsigned char) buffer[3]];
	}
}

void CpuAES::invMixColumns(char *bytes)
{
	char buffer[4];
	for(int i=0; i<4; ++i)
	{
		memcpy(buffer, bytes + i * 4, 4);

		bytes[i * 4] = MUL0E[(unsigned char) buffer[0]] ^ MUL0B[(unsigned char) buffer[1]] ^
				MUL0D[(unsigned char) buffer[2]] ^ MUL09[(unsigned char) buffer[3]];
		bytes[i * 4 + 1] = MUL09[(unsigned char) buffer[0]] ^ MUL0E[(unsigned char) buffer[1]] ^
				MUL0B[(unsigned char) buffer[2]] ^ MUL0D[(unsigned char) buffer[3]];
		bytes[i * 4 + 2] = MUL0D[(unsigned char) buffer[0]] ^ MUL09[(unsigned char) buffer[1]] ^
				MUL0E[(unsigned char) buffer[2]] ^ MUL0B[(unsigned char) buffer[3]];
		bytes[i * 4 + 3] = MUL0B[(unsigned char) buffer[0]] ^ MUL0D[(unsigned char) buffer[1]] ^
				MUL09[(unsigned char) buffer[2]] ^ MUL0E[(unsigned char) buffer[3]];
	}
}

/**
 * Encrypt a block
 */
void CpuAES::encryptBlock(char *in, char *out)
{
	memcpy(out, in, 16);

	addRoundKey(out, 0);

	for(int i = 1; i < Nr; ++i)
	{
		subBytes(out);
		shiftRows(out);
		mixColumns(out);
		addRoundKey(out, i);
	}

	subBytes(out);
	shiftRows(out);
	addRoundKey(out, Nr);
}

/**
 * Encrypt a block
 */
void CpuAES::decryptBlock(char *in, char *out)
{
	memcpy(out, in, 16);

	addRoundKey(out, Nr);

	for(int i = Nr - 1; i > 0; --i)
	{
		invShiftRows(out);
		invSubBytes(out);
		addRoundKey(out, i);
		invMixColumns(out);
	}

	invShiftRows(out);
	invSubBytes(out);
	addRoundKey(out, 0);
}

float CpuAES::encryptFile(const char *inName, const char *outName)
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
		buffer[i] = new char[chunkSize + 16];
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

			int numBlocks = currentChunkSize / 16 + (pos == size);
			int newLength = numBlocks * 16;
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
				offset += 16;
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

float CpuAES::decryptFile(const char *inName, const char *outName)
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

	if ((size & 15) != 0)
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

			int numBlocks = currentChunkSize / 16;
			int offset = 0;

			for (int j = 0; j < numBlocks; ++j)
			{
				decryptBlock(buffer[i] + offset, buffer[i] + offset);
				offset += 16;
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
