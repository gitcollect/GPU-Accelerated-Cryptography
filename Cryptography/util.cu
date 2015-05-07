/*
 * cudaUtil.cu
 *
 *  Created on: 2015-4-14
 *      Author: Yuqing Guan
 */

#include "util.h"

void gpuCheckError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
