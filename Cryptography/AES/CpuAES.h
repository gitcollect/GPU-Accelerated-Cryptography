/*
 * CpuAES.h
 *
 *  Created on: 2015-4-12
 *      Author: Yuqing Guan
 *  Reference:
 *      https://github.com/SongLee24/Aes_and_Des/blob/master/Aes/Aes.cpp
 */

#ifndef CPUAES_H_
#define CPUAES_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include "AES.h"

namespace AES
{

class CpuAES: public AES
{
private:
	char w[240][4];

	void bytes2Hex(char *in, char *out, int length);
	void hex2Bytes(char *in, char *out, int length);

	void rol(char *array, int length, int shift);
	void xorBlock(char *in1, char *in2, char *out, int length);
	void subWord(char *bytes);

	void addRoundKey(char *bytes, int round);

	void subBytes(char *bytes);
	void invSubBytes(char *bytes);

	void shiftRows(char *bytes);
	void invShiftRows(char *bytes);

	void mixColumns(char *bytes);
	void invMixColumns(char *bytes);

	void encryptBlock(char *in, char *out);
	void decryptBlock(char *in, char *out);
public:
	virtual float setKey(const char *key);

	virtual float encryptFile(const char *inName, const char *outName);
	virtual float decryptFile(const char *inName, const char *outName);
};

}

#endif /* CPUAES_H_ */
