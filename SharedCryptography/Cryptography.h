/*
 * Cryptography.h
 *
 *  Created on: 2015-4-27
 *      Author: Yuqing Guan
 */

#ifndef CRYPTOGRAPHY_H_
#define CRYPTOGRAPHY_H_

/**
 * Abstract class for cryptographic algorithms
 */
class Cryptography
{
public:
	Cryptography();
	virtual ~Cryptography();

	virtual float encryptFile(const char *inName, const char *outName) = 0;
	virtual float decryptFile(const char *inName, const char *outName) = 0;
};

#endif /* CRYPTOGRAPHY_H_ */
