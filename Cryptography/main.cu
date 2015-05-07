/*
 * main.cu
 *
 *  Created on: 2015-4-27
 *      Author: Yuqing Guan
 */

#include "DES/CpuDES.h"
#include "DES/GpuDES.h"

#include "AES/CpuAES.h"
#include "AES/GpuAES.h"

#include "RSA/CpuRSA.h"
#include "RSA/GpuRSA.h"

void printUsage()
{
	printf("Usages:\n");
	printf("\t1.\tCryptography cpudes encrypt in_file out_file password\n");
	printf("\t2.\tCryptography cpudes decrypt in_file out_file password\n");
	printf("\t3.\tCryptography gpudes encrypt in_file out_file password\n");
	printf("\t4.\tCryptography gpudes decrypt in_file out_file password\n");
	printf("\t5.\tCryptography cpuaes encrypt in_file out_file password\n");
	printf("\t6.\tCryptography cpuaes decrypt in_file out_file password\n");
	printf("\t7.\tCryptography gpuaes encrypt in_file out_file password\n");
	printf("\t8.\tCryptography gpuaes decrypt in_file out_file password\n");
	printf(
			"\t9.\tCryptography cpursa generate public_key_file private_key_file\n");
	printf(
			"\t10.\tCryptography cpursa encrypt in_file out_file public_key_file\n");
	printf(
			"\t11.\tCryptography cpursa decrypt in_file out_file private_key_file\n");
	printf(
			"\t12.\tCryptography gpursa generate public_key_file private_key_file\n");
	printf(
			"\t13.\tCryptography gpursa encrypt in_file out_file public_key_file\n");
	printf(
			"\t14.\tCryptography gpursa decrypt in_file out_file private_key_file\n");
}

/**
 * Test algorithms
 */
int main(int argc, char **argv)
{
	DES::CpuDES cpuDES;
	DES::GpuDES gpuDES;

	AES::CpuAES cpuAES;
	AES::GpuAES gpuAES;

	RSA::CpuRSA cpuRSA;
	RSA::GpuRSA gpuRSA;

	if (argc < 5)
	{
		printUsage();
		return -1;
	}

	float result = 0;

	if (strcmp(argv[1], "cpudes") == 0)
	{
		DES::DES *des = &cpuDES;

		if (argc < 6)
		{
			printUsage();
			return -1;
		}

		if (strcmp(argv[2], "encrypt") == 0)
		{
			result = des->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += des->encryptFile(argv[3], argv[4]);
			}
		}
		else if (strcmp(argv[2], "decrypt") == 0)
		{
			result = des->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += des->decryptFile(argv[3], argv[4]);
			}
		}
		else
		{
			printUsage();
			return -1;
		}
	}
	else if (strcmp(argv[1], "gpudes") == 0)
	{
		DES::DES *des = &gpuDES;

		if (argc < 6)
		{
			printUsage();
			return -1;
		}

		if (strcmp(argv[2], "encrypt") == 0)
		{
			result = des->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += des->encryptFile(argv[3], argv[4]);
			}
		}
		else if (strcmp(argv[2], "decrypt") == 0)
		{
			result = des->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += des->decryptFile(argv[3], argv[4]);
			}
		}
		else
		{
			printUsage();
			return -1;
		}
	}
	else if (strcmp(argv[1], "cpuaes") == 0)
	{
		AES::AES *aes = &cpuAES;

		if (argc < 6)
		{
			printUsage();
			return -1;
		}

		if (strcmp(argv[2], "encrypt") == 0)
		{
			result = aes->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += aes->encryptFile(argv[3], argv[4]);
			}
		}
		else if (strcmp(argv[2], "decrypt") == 0)
		{
			result = aes->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += aes->decryptFile(argv[3], argv[4]);
			}
		}
		else
		{
			printUsage();
			return -1;
		}
	}
	else if (strcmp(argv[1], "gpuaes") == 0)
	{
		AES::AES *aes = &gpuAES;

		if (argc < 6)
		{
			printUsage();
			return -1;
		}

		if (strcmp(argv[2], "encrypt") == 0)
		{
			result = aes->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += aes->encryptFile(argv[3], argv[4]);
			}
		}
		else if (strcmp(argv[2], "decrypt") == 0)
		{
			result = aes->setKey(argv[5]);

			if (result > -1e-6)
			{
				result += aes->decryptFile(argv[3], argv[4]);
			}
		}
		else
		{
			printUsage();
			return -1;
		}
	}
	else if (strcmp(argv[1], "cpursa") == 0)
	{
		RSA::RSA *rsa = &cpuRSA;

		if (strcmp(argv[2], "generate") == 0)
		{
			result = rsa->generateKeys(argv[3], argv[4]);
		}
		else
		{
			if (argc < 6)
			{
				printUsage();
				return -1;
			}

			if (strcmp(argv[2], "encrypt") == 0)
			{
				rsa->loadPublicKey(argv[5]);

				if (result > -1e-6)
				{
					result += rsa->encryptFile(argv[3], argv[4]);
				}
			}
			else if (strcmp(argv[2], "decrypt") == 0)
			{
				rsa->loadPrivateKey(argv[5]);

				if (result > -1e-6)
				{
					result += rsa->decryptFile(argv[3], argv[4]);
				}
			}
			else
			{
				printUsage();
				return -1;
			}
		}
	}
	else if (strcmp(argv[1], "gpursa") == 0)
	{
		RSA::RSA *rsa = &gpuRSA;

		if (strcmp(argv[2], "generate") == 0)
		{
			result = rsa->generateKeys(argv[3], argv[4]);
		}
		else
		{
			if (argc < 6)
			{
				printUsage();
				return -1;
			}

			if (strcmp(argv[2], "encrypt") == 0)
			{
				rsa->loadPublicKey(argv[5]);

				if (result > -1e-6)
				{
					result += rsa->encryptFile(argv[3], argv[4]);
				}
			}
			else if (strcmp(argv[2], "decrypt") == 0)
			{
				rsa->loadPrivateKey(argv[5]);

				if (result > -1e-6)
				{
					result += rsa->decryptFile(argv[3], argv[4]);
				}
			}
			else
			{
				printUsage();
				return -1;
			}
		}
	}
	else
	{
		printUsage();
		return -1;
	}

	if (result < 0)
	{
		int intResult = (int) result;

		switch (intResult)
		{
		case KEY_ERROR:
			printf("Key Error\n");
			return -1;
		case TEXT_ERROR:
			printf("Input File Content Error\n");
			return -1;
		case IN_FILE_ERROR:
			printf("Read Input File Error\n");
			return -1;
		case OUT_FILE_ERROR:
			printf("Write Output File Error\n");
			return -1;
		case DEVICE_ERROR:
			printf("GPU Device Error\n");
			return -1;
		default:
			printf("Unknown Error\n");
			return -1;
		}
	}

	printf("Completed in %f milliseconds.\n", result);

	return 0;
}
