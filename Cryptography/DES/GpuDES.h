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
#include <cuda.h>

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
	GpuDES();
	virtual ~GpuDES();

	virtual float setKey(const char *key);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* GPUDES_H_ */
