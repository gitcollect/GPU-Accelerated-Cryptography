/*
 * CpuDES.h
 *
 *  Created on: 2015-4-11
 *      Author: Yuqing Guan
 *
 *  Reference:
 *      http://www.cnblogs.com/imapla/archive/2012/09/07/2674788.html
 */

#ifndef CPUDES_H_
#define CPUDES_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "DES.h"

namespace DES
{

class CpuDES : public DES
{
private:
	bool subkey[16][48];

	void bytes2Bits(const char *in, bool *out, int length);
	void bits2Bytes(bool *in, char *out, int length);

	void hex2Bits(char *in, bool *out, int length);
	void bits2Hex(bool *in, char *out, int length);

	void permute(bool *in, bool *out, const char *table, int length);
	void rol(bool *array, int length, int shift);
	void xorBlock(bool *in1, bool *in2, bool *out, int length);

	void feistel(bool *array, int round);

	void encryptBlock(char *in, char *out);
	void decryptBlock(char *in, char *out);
public:
	virtual float setKey(const char *key);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* CPUDES_H_ */
