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
	GpuAES();
	virtual ~GpuAES();
	virtual float setKey(const char *key);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* GPUAES_H_ */
