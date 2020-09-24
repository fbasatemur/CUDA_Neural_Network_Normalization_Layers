#include "CpuGpuMat.h"

#ifdef __cplusplus									
extern "C"
#endif // __cplusplus

void gpuNormalizeAndShift(struct CpuGpuMat* cgMat, float mean, float variance, float gamma, float beta, float epsilon);