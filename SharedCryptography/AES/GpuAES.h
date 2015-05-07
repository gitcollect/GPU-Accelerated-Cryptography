/*
 * GpuAES.h
 *
 *  Created on: 2015-4-13
 *      Author: Yuqing Guan
 */

#ifndef GPUAES_H_
#define GPUAES_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "AES.h"

namespace AES
{

class GpuAES: public AES
{
private:
	int maxThreadY, maxBlockX;
	bool deviceReady;

	char *w;
public:
	char (*d_SBOX)[16], (*d_INV_SBOX)[16];
	char (*d_RCON)[4];

	char *d_MUL02, *d_MUL03, *d_MUL09;
	char *d_MUL0B, *d_MUL0D, *d_MUL0E;

	GpuAES *d_this;

	GpuAES();
	virtual ~GpuAES();
	virtual float setKey(const char *key);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* GPUAES_H_ */
