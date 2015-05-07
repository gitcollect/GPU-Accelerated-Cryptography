/*
 * GpuDES.h
 *
 *  Created on: 2015-4-13
 *      Author: Yuqing Guan
 */

#ifndef GPUDES_H_
#define GPUDES_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "DES.h"

namespace DES
{

class GpuDES: public DES
{
private:
	int maxThreadY, maxBlockX;
	bool deviceReady;

	bool *subkey;
public:
	char *d_IP, *d_FP;
	char *d_PC1, *d_PC2;
	char *d_SHIFT;
	char *d_E, *d_P;
	char (*d_SBOX)[4][16];

	GpuDES *d_this;

	GpuDES();
	virtual ~GpuDES();

	virtual float setKey(const char *key);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* GPUDES_H_ */
