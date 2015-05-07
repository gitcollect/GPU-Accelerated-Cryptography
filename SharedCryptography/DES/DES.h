/*
 * DES.h
 *
 *  Created on: 2015-4-12
 *      Author: Yuqing Guan
 */

#ifndef DES_H_
#define DES_H_

#include "../SymmetricCryptography.h"
#include "../util.h"

namespace DES
{

/**
 * Abstract class for DES
 */
class DES: public SymmetricCryptography
{
protected:
	bool keyReady;
public:
	static const char IP[64], FP[64];
	static const char PC1[56], PC2[48];
	static const char SHIFT[16];
	static const char E[48], P[32];
	static const char SBOX[8][4][16];

	static int chunkSize, maxStoredChunks;

	DES();
	virtual ~DES();

	virtual float setKey(const char *key) = 0;

	virtual float encryptFile(const char *inName, const char *outName) = 0;
	virtual float decryptFile(const char *inName, const char *outName) = 0;
};

}

#endif /* DES_H_ */
