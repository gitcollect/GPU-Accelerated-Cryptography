/*
 * util.h
 *
 *  Created on: 2015-4-14
 *      Author: Yuqing Guan
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <cstdio>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

// Errors

#define KEY_ERROR -1
#define TEXT_ERROR -2

#define IN_FILE_ERROR -3
#define OUT_FILE_ERROR -4

#define DEVICE_ERROR -5

#define CLOCK_PER_MILL (CLOCKS_PER_SEC / 1000.0)

#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__))

void gpuCheckError(cudaError_t err, const char *file, int line);

#endif /* UTIL_H_ */
