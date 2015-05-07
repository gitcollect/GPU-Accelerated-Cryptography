/*
 * RSA.h
 *
 *  Created on: 2015-4-23
 *      Author: Yuqing Guan
 */

#ifndef RSA_H_
#define RSA_H_

#include <cuda.h>

#include "../Cryptography.h"
#include "../util.h"

namespace RSA
{

#define SMALL_PRIMES_COUNT 168
#define DEFAULT_E 65537

/**
 * Abstract class for RSA
 */
class RSA: public Cryptography
{
protected:
	static const int SMALL_PRIMES[SMALL_PRIMES_COUNT];
	int ullSize, arrSize, halfArrSize;

	bool publicKeyReady, privateKeyReady;

public:
	static int chunkSize, maxStoredChunks;

	RSA();
	virtual ~RSA();

	virtual void loadPublicKey(const char *publicKey) = 0;
	virtual void loadPrivateKey(const char *privateKey) = 0;

	virtual float generateKeys(const char *publicKey,
			const char *privateKey) = 0;

	virtual float encryptFile(const char *inName, const char *outName) = 0;
	virtual float decryptFile(const char *inName, const char *outName) = 0;

	__host__ __device__ bool getBit(unsigned long long *in, int index);

	__host__ __device__ int length(unsigned long long *in);
	__host__ __device__ int bitLength(unsigned long long *in);

	__host__ __device__ int cmp(unsigned long long *in1,
			unsigned long long *in2);
	__host__ __device__ bool zero(unsigned long long *in);

	__host__ __device__ void print(unsigned long long *in, int length);
};

} /* namespace RSA */
#endif /* RSA_H_ */
