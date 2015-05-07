/*
 * CpuRSA.cpp
 *
 *  Created on: 2015-4-18
 *      Author: Yuqing Guan
 */

#include "CpuRSA.h"

namespace RSA
{

void CpuRSA::add(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int inLen1 = length(in1);
	int inLen2 = length(in2);

	int len = std::min(std::max(inLen1, inLen2) + 1, 32);

	unsigned long long oldIn1;

	bool carry[32];
	memset(carry, 0, 32);

	for (int i = 0; i < len; ++i)
	{
		oldIn1 = in1[i];
		out[i] = oldIn1 + in2[i];
		carry[i] = out[i] < oldIn1;
	}

	for (int i = 1; i < len; ++i)
	{
		if (carry[i - 1])
		{
			++out[i];

			if (out[i] == 0)
			{
				carry[i] = 1;
			}
		}
	}

	if (32 > len)
	{
		memset(out + len, 0, ullSize * (32 - len));
	}
}

void CpuRSA::sub(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int inLen1 = length(in1);
	int inLen2 = length(in2);

	int len = std::max(inLen1, inLen2);

	unsigned long long oldIn1;

	bool carry[32];
	memset(carry, 0, 32);

	for (int i = 0; i < len; ++i)
	{
		oldIn1 = in1[i];
		out[i] = oldIn1 - in2[i];
		carry[i] = out[i] > oldIn1;
	}

	for (int i = 1; i < len; ++i)
	{
		if (carry[i - 1])
		{
			if (out[i] == 0)
			{
				carry[i] = 1;
			}

			--out[i];
		}
	}

	if (32 > len)
	{
		memset(out + len, 0, ullSize * (32 - len));
	}
}

void CpuRSA::mul(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	int inLen1 = length(in1);
	int inLen2 = length(in2);

	int len = std::min(inLen1 + inLen2, 32);

	unsigned long long buffer[32], carry1[33], carry2[33], cur1[32], cur2[32];
	unsigned long long high1, high2, low1, low2, middle1, middle2;

	int k;

	memset(buffer, 0, arrSize);

	for (int i = 0; i < inLen1 && i < len; ++i)
	{
		memset(cur1, 0, arrSize);
		memset(cur2, 0, arrSize);

		memset(carry1, 0, arrSize + ullSize);
		memset(carry2, 0, arrSize + ullSize);

		high1 = in1[i] >> 32;
		low1 = in1[i] & 0xFFFFFFFF;

		for (int j = 0; j < inLen2 && i + j < len; ++j)
		{
			high2 = in2[j] >> 32;
			low2 = in2[j] & 0xFFFFFFFF;

			middle1 = low1 * high2;
			middle2 = low2 * high1;

			k = i + j;

			carry1[k + 1] = high1 * high2;
			carry2[k + 1] = (middle1 >> 32) + (middle2 >> 32)
					+ (((middle1 & 0xFFFFFFFF) + (middle2 & 0xFFFFFFFF)) >> 32);
			cur1[k] = low1 * low2;
			cur2[k] = (middle1 + middle2) << 32;
		}

		add(buffer, cur1, buffer);
		add(buffer, cur2, buffer);

		add(buffer, carry1, buffer);
		add(buffer, carry2, buffer);
	}

	memcpy(out, buffer, arrSize);
}

void CpuRSA::div(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out1, unsigned long long *out2)
{
	int inLen1 = length(in1);
	int inLen2 = length(in2);

	int cmpResult, diff;
	bool firstEnough, firstBinary;

	unsigned long long l, r, m;
	unsigned long long quotient[32], remainder[32], tmp[32];

	memset(quotient, 0, arrSize);
	memcpy(remainder, in1, arrSize);

	while (inLen1 >= inLen2)
	{
		diff = inLen1 - inLen2;
		firstEnough = quotient[diff] == 0
				&& remainder[inLen1 - 1] >= in2[inLen2 - 1];

		if (firstEnough)
		{
			l = 0;
			r = remainder[inLen1 - 1] / in2[inLen2 - 1];

			memset(tmp, 0, arrSize);
			tmp[diff] = m = l + ((r - l + 1) >> 1);

			if (r == 0xFFFFFFFFFFFFFFFF)
			{
				tmp[diff] = m = 0x8000000000000000;
				firstBinary = true;
			}

			while (l <= r && (firstBinary || r - l != 0xFFFFFFFFFFFFFFFF))
			{
				mul(tmp, in2, tmp);
				cmpResult = cmp(remainder, tmp);

				if (cmpResult >= 0)
				{
					quotient[diff] = m;
					l = m + 1;
				}
				else
				{
					r = m - 1;
				}

				memset(tmp, 0, arrSize);
				tmp[diff] = m = l + ((r - l + 1) >> 1);

				firstBinary = false;
			}

			if (quotient[diff] > 0)
			{
				memset(tmp, 0, arrSize);
				tmp[diff] = quotient[diff];

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

			memset(tmp, 0, arrSize);
			tmp[diff] = m = 0x8000000000000000;
			firstBinary = true;

			while (l <= r && (firstBinary || r - l != 0xFFFFFFFFFFFFFFFF))
			{
				mul(tmp, in2, tmp);
				cmpResult = cmp(remainder, tmp);

				if (cmpResult >= 0)
				{
					quotient[diff] = m;
					l = m + 1;
				}
				else
				{
					r = m - 1;
				}

				memset(tmp, 0, arrSize);
				tmp[diff] = m = l + ((r - l + 1) >> 1);

				firstBinary = false;
			}

			memset(tmp, 0, arrSize);
			tmp[diff] = quotient[diff];

			mul(tmp, in2, tmp);
			sub(remainder, tmp, remainder);
		}

		inLen1 = length(remainder);
	}

	if (out1 != NULL)
	{
		memcpy(out1, quotient, arrSize);
	}

	if (out2 != NULL)
	{
		memcpy(out2, remainder, arrSize);
	}
}

/**
 * Shift right
 */
void CpuRSA::shr(unsigned long long *in, unsigned long long *out, int offset)
{
	unsigned long long buffer[32];
	memset(buffer, 0, arrSize);

	int eleOffset = offset >> 6;
	int bitOffset = offset & 63;
	int invBitOffset = 64 - bitOffset;

	for (int i = 0; i < 32 - eleOffset; ++i)
	{
		buffer[i] = in[i + eleOffset] >> bitOffset;

		if (invBitOffset < 64 && i + eleOffset < 31)
		{
			buffer[i] |= in[i + eleOffset + 1] << invBitOffset;
		}
	}

	memcpy(out, buffer, arrSize);
}

/**
 * Random one unsigned long long
 */
unsigned long long CpuRSA::randomUll()
{
	unsigned long long high, low;

	high = rand();
	low = rand();

	return (high << 32) | low;
}

/**
 * Random big integer
 */
void CpuRSA::random(unsigned long long *in, int len)
{
	memset(in, 0, arrSize);

	for (int i = 0; i < len; ++i)
	{
		in[i] = randomUll();
	}

	while (in[len - 1] == 0)
	{
		in[len - 1] = randomUll();
	}
}

/**
 * Reference: http://www.di-mgt.com.au/euclidean.html
 */
void CpuRSA::extGcd(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	unsigned long long u1[32], u3[32], v1[32], v3[32], t1[32], t3[32], q[32];
	bool iter;

	memset(u1, 0, arrSize);
	memset(v1, 0, arrSize);

	memcpy(u3, in1, arrSize);
	memcpy(v3, in2, arrSize);

	u1[0] = 1;
	iter = 1;

	while (!zero(v3))
	{
		div(u3, v3, q, t3);
		mul(q, v1, q);
		add(u1, q, t1);

		memcpy(u1, v1, arrSize);
		memcpy(v1, t1, arrSize);
		memcpy(u3, v3, arrSize);
		memcpy(v3, t3, arrSize);

		iter = !iter;
	}

	if (!iter)
	{
		sub(in2, u1, out);
	}
	else
	{
		memcpy(out, u1, arrSize);
	}
}

/**
 * Extended Euclidean method to find modular multiplicative inverse
 * Add two 2048-bit integers and shift right by 1024 bits and preserve the bits higher than 1024
 */
void CpuRSA::addShift(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *out)
{
	unsigned long long low1[32], low2[32], high1[32], high2[32];

	memset(low1 + 16, 0, halfArrSize);
	memset(low2 + 16, 0, halfArrSize);

	memset(high1 + 16, 0, halfArrSize);
	memset(high2 + 16, 0, halfArrSize);

	memcpy(low1, in1, halfArrSize);
	memcpy(low2, in2, halfArrSize);

	memcpy(high1, in1 + 16, halfArrSize);
	memcpy(high2, in2 + 16, halfArrSize);

	add(low1, low2, low1);
	memcpy(low1, low1 + 16, halfArrSize);
	memset(low1 + 16, 0, halfArrSize);

	add(high1, high2, high1);
	add(high1, low1, out);
}

/**
 * Montgomery modular multiplication
 * Reference: http://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
 */
void CpuRSA::powMod(unsigned long long *in1, unsigned long long *in2,
		unsigned long long *in3, unsigned long long *out,
		unsigned long long *invR, unsigned long long *invP)
{
	int bitLen = bitLength(in2);

	unsigned long long base[32];
	mul(in1, R, base);
	div(base, in3, NULL, base);

	unsigned long long buffer[32];
	memcpy(buffer, R, arrSize);
	div(buffer, in3, NULL, buffer);

	unsigned long long k[32];

	for (int i = 0; i < bitLen; ++i)
	{
		if (getBit(in2, i))
		{
			mul(base, buffer, buffer);

			memset(k + 16, 0, halfArrSize);
			memcpy(k, buffer, halfArrSize);

			mul(k, invP, k);
			memset(k + 16, 0, halfArrSize);
			sub(R, k, k);

			mul(k, in3, k);
			addShift(buffer, k, buffer);

			while (length(buffer) > 16)
			{
				sub(buffer, in3, buffer);
			}
		}

		mul(base, base, base);

		memset(k + 16, 0, halfArrSize);
		memcpy(k, base, halfArrSize);

		mul(k, invP, k);
		memset(k + 16, 0, halfArrSize);
		sub(R, k, k);

		mul(k, in3, k);
		addShift(base, k, base);

		while (length(base) > 16)
		{
			sub(base, in3, base);
		}
	}

	mul(buffer, invR, buffer);
	div(buffer, in3, NULL, out);
}

/**
 * Miller–Rabin primality test
 * Reference: My Homework for 'Introduction to Information Security' in Fall 2013, Peking University
 */
bool CpuRSA::millerRabin(unsigned long long *in, unsigned long long *sub1)
{
	unsigned long long a[32], m[32], z[32];
	unsigned long long tmp[32];

	memcpy(m, sub1, arrSize);
	memset(tmp, 0, arrSize);

	unsigned long long invR[32];
	extGcd(R, in, invR);

	unsigned long long invP[32];
	extGcd(in, R, invP);

	int s, randLen;

	bool passOneTest;

	s = 0;
	while (!getBit(sub1, s))
	{
		++s;
	}

	shr(m, m, s);

	int len = length(in);

	for (int i = 0; i < 32; ++i)
	{
		tmp[0] = 2;

		do
		{
			randLen = rand() % len + 1;
			random(a, randLen);
			add(a, tmp, a);

			if (sub1[len - 1] < 0xFFFFFFFFFFFFFFFF)
			{
				a[len - 1] %= (sub1[len - 1] + 1);
			}
		}
		while (cmp(sub1, a) <= 0);

		powMod(a, m, in, z, invR, invP);

		tmp[0] = 1;
		if (cmp(z, tmp) == 0)
		{
			continue;
		}

		tmp[0] = 2;

		passOneTest = false;
		for (int j = 0; j < s; ++j)
		{
			if (cmp(z, sub1) == 0)
			{
				passOneTest = true;
				break;
			}

			powMod(z, tmp, in, z, invR, invP);
		}

		if (!passOneTest)
		{
			return false;
		}
	}

	return true;
}

/**
 * Generate random big prime numbers
 */
void CpuRSA::randomPrime(unsigned long long *out, unsigned long long *sub1)
{
	unsigned long long remainder[32];
	bool foundPrime = false;

	unsigned long long tmp[32];
	memset(tmp, 0, arrSize);

	while (!foundPrime)
	{
		random(out, 8);
		out[0] |= 1;
		out[7] |= 0x8000000000000000;

		tmp[0] = 1;
		sub(out, tmp, sub1);

		tmp[0] = DEFAULT_E;
		div(sub1, tmp, NULL, remainder);

		if (zero(remainder))
		{
			continue;
		}

		// Check with small prime numbers

		foundPrime = true;
		for (int i = 0; i < SMALL_PRIMES_COUNT; ++i)
		{
			tmp[0] = SMALL_PRIMES[i];
			div(out, tmp, NULL, remainder);

			if (zero(remainder))
			{
				foundPrime = false;
				break;
			}
		}

		// Miller-Rabin tests

		if (foundPrime)
		{
			foundPrime = millerRabin(out, sub1);
		}
	}
}

/**
 * Generate public and private key
 */
float CpuRSA::generateKeys(const char *publicKey, const char *privateKey)
{
	clock_t start = clock();

	unsigned long long P[32], Q[32], P_1[32], Q_1[32], r[32];

	memset(E, 0, arrSize);
	E[0] = DEFAULT_E;

	// Generate big prime numbers P, Q

	randomPrime(P, P_1);

	do
	{
		randomPrime(Q, Q_1);
	}
	while (cmp(P, Q) == 0);

	// Compute N = PQ, e = 65537, ed % (P - 1)(Q - 1) = 1

	mul(P, Q, N);
	mul(P_1, Q_1, r);

	extGcd(E, r, D);

	publicKeyReady = privateKeyReady = true;

	// Output public and private key

	FILE *outFile = fopen(publicKey, "wb");
	if (outFile == NULL)
	{
		return OUT_FILE_ERROR;
	}

	if (fwrite(N, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}
	if (fwrite(E, ullSize, 16, outFile) != 16)
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

	if (fwrite(N, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}

	if (fwrite(D, ullSize, 16, outFile) != 16)
	{
		fclose(outFile);
		return OUT_FILE_ERROR;
	}

	fclose(outFile);

	return (clock() - start) / CLOCK_PER_MILL;
}

void CpuRSA::loadPublicKey(const char *publicKey)
{
	FILE *inFile = fopen(publicKey, "rb");
	if (inFile == NULL)
	{
		publicKeyReady = false;
	}

	memset(N + 16, 0, halfArrSize);
	if (fread(N, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		publicKeyReady = false;
	}

	memset(E + 16, 0, halfArrSize);
	if (fread(E, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		publicKeyReady = false;
	}

	fclose(inFile);
	publicKeyReady = true;
}

void CpuRSA::loadPrivateKey(const char *privateKey)
{
	FILE *inFile = fopen(privateKey, "rb");
	if (inFile == NULL)
	{
		privateKeyReady = false;
	}

	memset(N + 16, 0, halfArrSize);
	if (fread(N, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		privateKeyReady = false;
	}

	memset(D + 16, 0, halfArrSize);
	if (fread(D, ullSize, 16, inFile) != 16)
	{
		fclose(inFile);
		privateKeyReady = false;
	}

	fclose(inFile);
	privateKeyReady = true;
}

/**
 *  m ^ e % N = c
 */
void CpuRSA::encryptBlock(char *in, char *out)
{
	unsigned long long buffer[32];

	memset(buffer, 0, arrSize);
	memcpy(buffer, in, halfArrSize - 1);

	unsigned long long invR[32];
	extGcd(R, N, invR);

	unsigned long long invP[32];
	extGcd(N, R, invP);

	powMod(buffer, E, N, buffer, invR, invP);

	memcpy(out, buffer, halfArrSize);
}

/**
 * c ^ d % N = m
 */
void CpuRSA::decryptBlock(char *in, char *out)
{
	unsigned long long buffer[32];

	memset(buffer, 0, arrSize);
	memcpy(buffer, in, halfArrSize);

	unsigned long long invR[32];
	extGcd(R, N, invR);

	unsigned long long invP[32];
	extGcd(N, R, invP);

	powMod(buffer, D, N, buffer, invR, invP);

	memcpy(out, buffer, halfArrSize - 1);
}

CpuRSA::CpuRSA()
{
	memset(R, 0, arrSize);
	R[16] = 1;
}

CpuRSA::~CpuRSA()
{

}

float CpuRSA::encryptFile(const char *inName, const char *outName)
{
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

	clock_t start = clock();

	char **inBuffer = new char*[maxStoredChunks];
	char **outBuffer = new char*[maxStoredChunks];

	int blockPerChunk = chunkSize / 127;

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		inBuffer[i] = new char[chunkSize + 127];
		outBuffer[i] = new char[chunkSize + 128 + blockPerChunk];
	}

	fseek(inFile, 0, SEEK_SET);

	size_t pos = 0, currentInChunkSize, currentOutChunkSize;

	while (pos < size)
	{
		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentInChunkSize = chunkSize;

			if (pos + currentInChunkSize > size)
			{
				currentInChunkSize = size - pos;
			}

			if (fread(inBuffer[i], 1, currentInChunkSize, inFile)
					!= currentInChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentInChunkSize;

			int numBlocks = currentInChunkSize / 127 + (pos == size);
			int newLength = numBlocks * 127;
			int padding = newLength - currentInChunkSize;

			if (padding > 0)
			{
				memset(inBuffer[i] + currentInChunkSize, padding, padding);
				currentInChunkSize = newLength;
			}

			currentOutChunkSize = numBlocks * 128;

			int inOffset = 0, outOffset = 0;

			for (int j = 0; j < numBlocks; ++j)
			{
				encryptBlock(inBuffer[i] + inOffset, outBuffer[i] + outOffset);

				inOffset += 127;
				outOffset += 128;
			}

			if (fwrite(outBuffer[i], 1, currentOutChunkSize, outFile)
					!= currentOutChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return OUT_FILE_ERROR;
			}
		}
	}

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		delete inBuffer[i];
		delete outBuffer[i];
	}

	delete inBuffer;
	delete outBuffer;

	fclose(inFile);
	fclose(outFile);

	return (clock() - start) / CLOCK_PER_MILL;
}

float CpuRSA::decryptFile(const char *inName, const char *outName)
{
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

	if ((size & 127) != 0)
	{
		fclose(inFile);
		fclose(outFile);

		return TEXT_ERROR;
	}

	clock_t start = clock();

	char **inBuffer = new char*[maxStoredChunks];
	char **outBuffer = new char*[maxStoredChunks];

	int blockPerChunk = chunkSize / 127;

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		inBuffer[i] = new char[chunkSize + blockPerChunk];
		outBuffer[i] = new char[chunkSize];
	}

	fseek(inFile, 0, SEEK_SET);

	size_t pos = 0, currentInChunkSize, currentOutChunkSize;

	while (pos < size)
	{
		for (int i = 0; i < maxStoredChunks && pos < size; ++i)
		{
			currentInChunkSize = chunkSize + blockPerChunk;
			if (pos + currentInChunkSize > size)
			{
				currentInChunkSize = size - pos;
			}

			if (fread(inBuffer[i], 1, currentInChunkSize, inFile)
					!= currentInChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return IN_FILE_ERROR;
			}

			pos += currentInChunkSize;

			int numBlocks = currentInChunkSize / 128;
			int inOffset = 0, outOffset = 0;

			currentOutChunkSize = numBlocks * 127;

			for (int j = 0; j < numBlocks; ++j)
			{
				decryptBlock(inBuffer[i] + inOffset, outBuffer[i] + outOffset);

				inOffset += 128;
				outOffset += 127;
			}

			if (pos == size)
			{
				currentOutChunkSize -= outBuffer[i][currentOutChunkSize - 1];
			}

			if (fwrite(outBuffer[i], 1, currentOutChunkSize, outFile)
					!= currentOutChunkSize)
			{
				fclose(inFile);
				fclose(outFile);

				return OUT_FILE_ERROR;
			}
		}
	}

	for (int i = 0; i < maxStoredChunks; ++i)
	{
		delete inBuffer[i];
		delete outBuffer[i];
	}

	delete inBuffer;
	delete outBuffer;

	fclose(inFile);
	fclose(outFile);

	return (clock() - start) / CLOCK_PER_MILL;
}

}
