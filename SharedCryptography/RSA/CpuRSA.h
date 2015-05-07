/*
 * CpuRSA.h
 *
 *  Created on: 2015-4-18
 *      Author: Yuqing Guan
 */

#ifndef CPURSA_H_
#define CPURSA_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "RSA.h"

namespace RSA
{

class CpuRSA: public RSA
{
private:
	unsigned long long E[32], D[32], N[32], R[32];

	void add(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);
	void sub(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);
	void mul(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);
	void div(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out1, unsigned long long *out2);
	void shr(unsigned long long *in, unsigned long long *out, int offset);

	unsigned long long randomUll();
	void random(unsigned long long *in, int len);

	void extGcd(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);

	void addShift(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *out);

	void powMod(unsigned long long *in1, unsigned long long *in2,
			unsigned long long *in3, unsigned long long *out,
			unsigned long long *invR, unsigned long long *invN0);

	bool millerRabin(unsigned long long *in, unsigned long long *sub1);
	void randomPrime(unsigned long long *out, unsigned long long *sub1);

	void encryptBlock(char *in, char *out);
	void decryptBlock(char *in, char *out);
public:
	CpuRSA();
	virtual ~CpuRSA();

	virtual void loadPublicKey(const char *publicKey);
	virtual void loadPrivateKey(const char *privateKey);

	virtual float generateKeys(const char *publicKey, const char *privateKey);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* CPURSA_H_ */
