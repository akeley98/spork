#include "xgemm_Sm90_n128.cuh"


__launch_bounds__(384, 1)
__global__ void
exo_deviceFunction0_xgemm_Sm90_wgmma_n128(__grid_constant__ const struct exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n128 exo_deviceArgs)
{
  extern __shared__ char exo_smem[];
  exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceSetup(exo_smem, exo_deviceArgs);
  exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceMainLoop(exo_smem, exo_deviceArgs);
}

void
exo_cudaLaunch0_xgemm_Sm90_wgmma_n128(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n128 exo_deviceArgs)
{
  exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_cudaLaunch(exo_cudaStream, exo_deviceArgs);
}


