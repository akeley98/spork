#include "edited.cuh"


__launch_bounds__(384, 1)
__global__ void
exo_deviceFunction0_edited_Sm90_wgmma(__grid_constant__ const struct exo_CudaDeviceArgs0_edited_Sm90_wgmma exo_deviceArgs)
{
  extern __shared__ char exo_smem[];
  exo_Cuda0_edited_Sm90_wgmma::exo_deviceSetup(exo_smem, exo_deviceArgs);
  exo_Cuda0_edited_Sm90_wgmma::exo_deviceMainLoop(exo_smem, exo_deviceArgs);
}

void
exo_cudaLaunch0_edited_Sm90_wgmma(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs0_edited_Sm90_wgmma exo_deviceArgs)
{
  exo_Cuda0_edited_Sm90_wgmma::exo_cudaLaunch(exo_cudaStream, exo_deviceArgs);
}


