// If apple, use Accelerate
#if defined(USE_MKL)
#   include <mkl_cblas.h>
#elif defined __APPLE__
#   include <Accelerate/Accelerate.h>
#else
#   include <cblas.h>
#endif