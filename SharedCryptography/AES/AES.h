/*
 * AES.h
 *
 *  Created on: 2015-4-12
 *      Author: Yuqing Guan
 */

#ifndef AES_H_
#define AES_H_

#include "../SymmetricCryptography.h"
#include "../util.h"

namespace AES
{

/**
 * Abstract class for AES
 */
class AES: public SymmetricCryptography
{
protected:
	int Nk, Nr;
	bool keyReady;
public:
	static const char SBOX[16][16], INV_SBOX[16][16];
	static const char RCON[16][4];

	static const char MUL02[256], MUL03[256], MUL09[256];
	static const char MUL0B[256], MUL0D[256], MUL0E[256];

	static int chunkSize, maxStoredChunks;

	AES();
	virtual ~AES();

	virtual float setKey(const char *key) = 0;

	virtual float encryptFile(const char *inName, const char *outName) = 0;
	virtual float decryptFile(const char *inName, const char *outName) = 0;
};

}

#endif /* AES_H_ */
