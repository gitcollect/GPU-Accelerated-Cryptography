/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class edu_columbia_gpu11_CuWrapper */

#ifndef _Included_edu_columbia_gpu11_CuWrapper
#define _Included_edu_columbia_gpu11_CuWrapper
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     edu_columbia_gpu11_CuWrapper
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_columbia_gpu11_CuWrapper_init
  (JNIEnv *, jobject);

/*
 * Class:     edu_columbia_gpu11_CuWrapper
 * Method:    doAlgo
 * Signature: (IILjava/lang/String;Ljava/lang/String;Ljava/lang/String;)F
 */
JNIEXPORT jfloat JNICALL Java_edu_columbia_gpu11_CuWrapper_doAlgo
  (JNIEnv *, jobject, jint, jint, jstring, jstring, jstring);

/*
 * Class:     edu_columbia_gpu11_CuWrapper
 * Method:    undoAlgo
 * Signature: (IILjava/lang/String;Ljava/lang/String;Ljava/lang/String;)F
 */
JNIEXPORT jfloat JNICALL Java_edu_columbia_gpu11_CuWrapper_undoAlgo
  (JNIEnv *, jobject, jint, jint, jstring, jstring, jstring);

/*
 * Class:     edu_columbia_gpu11_CuWrapper
 * Method:    genRSA
 * Signature: (ILjava/lang/String;Ljava/lang/String;)F
 */
JNIEXPORT jfloat JNICALL Java_edu_columbia_gpu11_CuWrapper_genRSA
  (JNIEnv *, jobject, jint, jstring, jstring);

#ifdef __cplusplus
}
#endif
#endif
