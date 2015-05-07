/*
 * GpuRSA.h
 *
 *  Created on: 2015-4-23
 *      Author: Yuqing Guan
 */

#ifndef GPURSA_H_
#define GPURSA_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "RSA.h"

namespace RSA
{

class GpuRSA: public RSA
{
private:
	int maxBlockX;
	bool deviceReady;

	GpuRSA *d_this;

public:
	unsigned long long *E, *D, *N;
	curandState *states;

	__device__ void add(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);
	__device__ void sub(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);
	__device__ void mul(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);
	__device__ void div(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out1, unsigned long long *out2);

	__device__ void shr(unsigned long long *in, unsigned long long *out,
			int offset);

	__device__ unsigned long long randomUll(curandState *states);
	__device__ void random(unsigned long long *in, int len,
			curandState *states);

	__device__ void extGcd(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);

	__device__ void addShift(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);

	__device__ void powMod(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *in3, unsigned long long *out,
			unsigned long long *invR, unsigned long long *invN0);

	void randomPrime(unsigned long long *out, unsigned long long *sub1,
			unsigned long long *z, unsigned long long *invR,
			unsigned long long *invP, int *d_s, bool *passTest,
			bool *d_isPrime);

	GpuRSA();
	virtual ~GpuRSA();

	virtual void loadPublicKey(const char *publicKey);
	virtual void loadPrivateKey(const char *privateKey);

	virtual float generateKeys(const char *publicKey, const char *privateKey);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

} /* namespace RSA */
#endif /* GPURSA_H_ */
