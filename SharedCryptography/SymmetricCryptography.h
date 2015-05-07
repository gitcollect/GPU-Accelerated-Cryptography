/*
 * SymmetricCryptography.h
 *
 *  Created on: 2015-4-27
 *      Author: Yuqing Guan
 */

#ifndef SYMMETRICCRYPTOGRAPHY_H_
#define SYMMETRICCRYPTOGRAPHY_H_

#include "Cryptography.h"

/**
 * Abstract class for symmetric cryptographic algorithms
 */
class SymmetricCryptography: public Cryptography
{
public:
	SymmetricCryptography();
	virtual ~SymmetricCryptography();

	virtual float setKey(const char *key) = 0;

	virtual float encryptFile(const char *inName, const char *outName) = 0;
	virtual float decryptFile(const char *inName, const char *outName) = 0;
};

#endif /* SYMMETRICCRYPTOGRAPHY_H_ */
