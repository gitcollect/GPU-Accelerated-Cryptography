/*
 * RSA.cpp
 *
 *  Created on: 2015-4-23
 *      Author: Yuqing Guan
 */

#include "RSA.h"

namespace RSA
{

int RSA::chunkSize = 127 << 18;
int RSA::maxStoredChunks = 4;

/**
 * Prime numbers less than 1000
 */
const int RSA::SMALL_PRIMES[SMALL_PRIMES_COUNT] = { 2, 3, 5, 7, 11, 13, 17, 19,
		23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
		103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
		179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
		257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
		347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
		431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
		509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
		607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683,
		691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
		797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881,
		883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983,
		991, 997

};

/**
 * Get bit at specified position
 */
__host__ __device__ bool RSA::getBit(unsigned long long *in, int index)
{
	int eleIndex = index >> 6;
	int bitIndex = index & 63;

	return (in[eleIndex] >> bitIndex) & 1;
}

/**
 * Get unsigned long long length of 2048-bit integer
 */
__host__ __device__ int RSA::length(unsigned long long *in)
{
	int len = 32;

	while (len > 0 && in[len - 1] == 0)
	{
		--len;
	}

	return len;
}

/**
 * Get bit length of 2048-bit integer
 */
__host__ __device__ int RSA::bitLength(unsigned long long *in)
{
	int len = length(in) - 1;
	unsigned long long tmp = in[len];

	len <<= 6;

	while (tmp)
	{
		++len;
		tmp >>= 1;
	}

	return len;
}

/**
 * Compare 2048-bit integers
 */
__host__ __device__ int RSA::cmp(unsigned long long *in1, unsigned long long *in2)
{
	for (int i = 31; i >= 0; --i)
	{
		unsigned long long num1, num2;

		num1 = in1[i];
		num2 = in2[i];

		if (num1 != num2)
		{
			return num1 < num2 ? -1 : 1;
		}
	}

	return 0;
}

/**
 * Check whether a 2048-bit integer is zero
 */
__host__ __device__ bool RSA::zero(unsigned long long *in)
{
	for (int i = 0; i < 32; ++i)
	{
		if (in[i])
		{
			return false;
		}
	}

	return true;
}

/**
 * Print a big integer
 */
__host__ __device__ void RSA::print(unsigned long long *in, int length)
{
	printf("0x");
	for (int i = length - 1; i >= 0; --i)
	{
		printf("%016llX", in[i]);
	}
}

RSA::RSA()
{
	srand(time(NULL));
	publicKeyReady = privateKeyReady = false;

	ullSize = sizeof(unsigned long long);
	arrSize = ullSize * 32;
	halfArrSize = arrSize >> 1;
}

RSA::~RSA()
{

}

} /* namespace RSA */
