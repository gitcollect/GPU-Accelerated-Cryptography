/**
 * The JNI C++ file to embed our CUDA code to Java
 */

#include "edu_columbia_gpu11_CuWrapper.h"

#include "DES/CpuDES.h"
#include "DES/GpuDES.h"

#include "AES/CpuAES.h"
#include "AES/GpuAES.h"

#include "RSA/CpuRSA.h"
#include "RSA/GpuRSA.h"

#define ALG_DES 0
#define ALG_AES 1
#define ALG_RSA 2

#define DEV_CPU 0
#define DEV_GPU 1

// We must define the GPU constant memory here, otherwise they cannot be accessed in Java Virtual Machine

__constant__ char d_DES_IP[64], d_DES_FP[64];
__constant__ char d_DES_PC1[56], d_DES_PC2[48];
__constant__ char d_DES_SHIFT[16];
__constant__ char d_DES_E[48], d_DES_P[32];
__constant__ char d_DES_SBOX[8][4][16];

__constant__ char d_AES_SBOX[16][16], d_AES_INV_SBOX[16][16];
__constant__ char d_AES_RCON[16][4];

__constant__ char d_AES_MUL02[256], d_AES_MUL03[256], d_AES_MUL09[256];
__constant__ char d_AES_MUL0B[256], d_AES_MUL0D[256], d_AES_MUL0E[256];

__constant__ int d_RSA_SMALL_PRIMES[SMALL_PRIMES_COUNT];
__constant__ unsigned long long d_RSA_R[32];

DES::CpuDES cpuDES;
DES::GpuDES gpuDES;

AES::CpuAES cpuAES;
AES::GpuAES gpuAES;

RSA::CpuRSA cpuRSA;
RSA::GpuRSA gpuRSA;

/**
 * Copy the pointers to GPU constant memory to objects, then upload them to GPU's memory
 */JNIEXPORT void JNICALL Java_edu_columbia_gpu11_CuWrapper_init(JNIEnv *,
		jobject)
{
	// DES

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_IP, DES::DES::IP, sizeof(DES::DES::IP)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_FP, DES::DES::FP, sizeof(DES::DES::FP)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_PC1, DES::DES::PC1,
					sizeof(DES::DES::PC1)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_PC2, DES::DES::PC2,
					sizeof(DES::DES::PC2)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_SHIFT, DES::DES::SHIFT,
					sizeof(DES::DES::SHIFT)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_E, DES::DES::E, sizeof(DES::DES::E)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_P, DES::DES::P, sizeof(DES::DES::P)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_DES_SBOX, DES::DES::SBOX,
					sizeof(DES::DES::SBOX)));

	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_IP, d_DES_IP));
	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_FP, d_DES_FP));

	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_PC1, d_DES_PC1));
	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_PC2, d_DES_PC2));

	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuDES.d_SHIFT, d_DES_SHIFT));

	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_E, d_DES_E));
	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_P, d_DES_P));

	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuDES.d_SBOX, d_DES_SBOX));

	GPU_CHECKERROR(
			cudaMemcpy(gpuDES.d_this, &gpuDES, sizeof(DES::GpuDES),
					cudaMemcpyHostToDevice));

	// AES

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_SBOX, AES::AES::SBOX,
					sizeof(AES::AES::SBOX)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_INV_SBOX, AES::AES::INV_SBOX,
					sizeof(AES::AES::INV_SBOX)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_RCON, AES::AES::RCON,
					sizeof(AES::AES::RCON)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_MUL02, AES::AES::MUL02,
					sizeof(AES::AES::MUL02)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_MUL03, AES::AES::MUL03,
					sizeof(AES::AES::MUL03)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_MUL09, AES::AES::MUL09,
					sizeof(AES::AES::MUL09)));

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_MUL0B, AES::AES::MUL0B,
					sizeof(AES::AES::MUL0B)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_MUL0D, AES::AES::MUL0D,
					sizeof(AES::AES::MUL0D)));
	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_AES_MUL0E, AES::AES::MUL0E,
					sizeof(AES::AES::MUL0E)));

	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuAES.d_SBOX, d_AES_SBOX));
	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_INV_SBOX,
					d_AES_INV_SBOX));

	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuAES.d_RCON, d_AES_RCON));

	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_MUL02, d_AES_MUL02));
	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_MUL03, d_AES_MUL03));
	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_MUL09, d_AES_MUL09));

	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_MUL0B, d_AES_MUL0B));
	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_MUL0D, d_AES_MUL0D));
	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuAES.d_MUL0E, d_AES_MUL0E));

	GPU_CHECKERROR(
			cudaMemcpy(gpuAES.d_this, &gpuAES, sizeof(AES::GpuAES),
					cudaMemcpyHostToDevice));

	// RSA

	unsigned long long R[32];

	memset(R, 0, sizeof(R));
	R[16] = 1;

	GPU_CHECKERROR(
			cudaMemcpyToSymbol(d_RSA_SMALL_PRIMES, RSA::RSA::SMALL_PRIMES,
					sizeof(RSA::RSA::SMALL_PRIMES)));
	GPU_CHECKERROR(cudaMemcpyToSymbol(d_RSA_R, R, sizeof(R)));

	GPU_CHECKERROR(
			cudaGetSymbolAddress((void ** ) &gpuRSA.d_SMALL_PRIMES,
					d_RSA_SMALL_PRIMES));
	GPU_CHECKERROR(cudaGetSymbolAddress((void ** ) &gpuRSA.d_R, d_RSA_R));

	GPU_CHECKERROR(
			cudaMemcpy(gpuRSA.d_this, &gpuRSA, sizeof(RSA::GpuRSA),
					cudaMemcpyHostToDevice));
}

 /**
  * Encrypt algorithms
  */
JNIEXPORT jfloat JNICALL Java_edu_columbia_gpu11_CuWrapper_doAlgo(JNIEnv *env,
		jobject, jint alg, jint dev, jstring jInName, jstring jOutName,
		jstring jArg4)
{
	if (dev != DEV_CPU && dev != DEV_GPU)
	{
		return WRONG_DEVICE;
	}

	float result;

	jboolean isCopy;

	const char *inName = env->GetStringUTFChars(jInName, &isCopy);
	const char *outName = env->GetStringUTFChars(jOutName, &isCopy);
	const char *arg4 = env->GetStringUTFChars(jArg4, &isCopy);

	DES::DES *des = NULL;
	AES::AES *aes = NULL;
	RSA::RSA *rsa = NULL;

	// Select an algorithm and a device

	switch (alg)
	{
	case ALG_DES:
		des = dev == DEV_CPU ? (DES::DES *) &cpuDES : (DES::DES *) &gpuDES;

		result = des->setKey(arg4);
		if (result > -1e-6)
		{
			result += des->encryptFile(inName, outName);
		}

		break;
	case ALG_AES:
		aes = dev == DEV_CPU ? (AES::AES *) &cpuAES : (AES::AES *) &gpuAES;

		result = aes->setKey(arg4);
		if (result > -1e-6)
		{
			result += aes->encryptFile(inName, outName);
		}

		break;
	case ALG_RSA:
		rsa = dev == DEV_CPU ? (RSA::RSA *) &cpuRSA : (RSA::RSA *) &gpuRSA;

		rsa->loadPublicKey(arg4);
		result = rsa->encryptFile(inName, outName);

		break;
	default:
		result = WRONG_ALGORITHM;
	}

	env->ReleaseStringUTFChars(jInName, inName);
	env->ReleaseStringUTFChars(jOutName, outName);
	env->ReleaseStringUTFChars(jArg4, arg4);

	return result;
}

/**
 * Decrypt algorithms
 */
JNIEXPORT jfloat JNICALL Java_edu_columbia_gpu11_CuWrapper_undoAlgo(JNIEnv *env,
		jobject, jint alg, jint dev, jstring jInName, jstring jOutName,
		jstring jArg4)
{
	if (dev != DEV_CPU && dev != DEV_GPU)
	{
		return WRONG_DEVICE;
	}

	float result;

	jboolean isCopy;

	const char *inName = env->GetStringUTFChars(jInName, &isCopy);
	const char *outName = env->GetStringUTFChars(jOutName, &isCopy);
	const char *arg4 = env->GetStringUTFChars(jArg4, &isCopy);

	DES::DES *des = NULL;
	AES::AES *aes = NULL;
	RSA::RSA *rsa = NULL;

	// Select an algorithm and a device

	switch (alg)
	{
	case ALG_DES:
		des = dev == DEV_CPU ? (DES::DES *) &cpuDES : (DES::DES *) &gpuDES;

		result = des->setKey(arg4);
		if (result > -1e-6)
		{
			result += des->decryptFile(inName, outName);
		}

		break;
	case ALG_AES:
		aes = dev == DEV_CPU ? (AES::AES *) &cpuAES : (AES::AES *) &gpuAES;

		result = aes->setKey(arg4);
		if (result > -1e-6)
		{
			result += aes->decryptFile(inName, outName);
		}

		break;
	case ALG_RSA:
		rsa = dev == DEV_CPU ? (RSA::RSA *) &cpuRSA : (RSA::RSA *) &gpuRSA;

		rsa->loadPrivateKey(arg4);
		result = rsa->decryptFile(inName, outName);

		break;
	default:
		result = WRONG_ALGORITHM;
	}

	env->ReleaseStringUTFChars(jInName, inName);
	env->ReleaseStringUTFChars(jOutName, outName);
	env->ReleaseStringUTFChars(jArg4, arg4);

	return result;
}

/**
 * Generate RSA key pair
 */
JNIEXPORT jfloat JNICALL Java_edu_columbia_gpu11_CuWrapper_genRSA(JNIEnv *env,
		jobject, jint dev, jstring jPublicKey, jstring jPrivateKey)
{
	jboolean isCopy;

	if (dev != DEV_CPU && dev != DEV_GPU)
	{
		return WRONG_DEVICE;
	}

	RSA::RSA *rsa =
			dev == DEV_CPU ? (RSA::RSA *) &cpuRSA : (RSA::RSA *) &gpuRSA;

	const char *publicKey = env->GetStringUTFChars(jPublicKey, &isCopy);
	const char *privateKey = env->GetStringUTFChars(jPrivateKey, &isCopy);

	float result = rsa->generateKeys(publicKey, privateKey);

	env->ReleaseStringUTFChars(jPublicKey, publicKey);
	env->ReleaseStringUTFChars(jPrivateKey, privateKey);

	return result;
}

