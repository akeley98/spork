// intra-thread issue
Sm80_cp_async_f32(smem[i], ...);
// asm volatile("cp.async.cg.shared.global {smem}");
accum += smem;


// inter-thread issue
smem[threadIdx.x] = gmem[threadIdx.x];
for (int i = 0; i < blockDim.x; ++i) {
    accum += smem[i];
}


// mixed issue (more interesting)
asm volatile("wgmma.mma_async... {wgmma_D} += {smem_A} {smem_B};");
cluster_sync();
// ... *DISTRACTION*
asm volatile("wgmma.wait_group.sync.aligned 0;");
asm volatile("barrier.cluster.arrive.aligned;");
gmem[...] = wgmma_D[...];
asm volatile("barrier.cluster.await.aligned;");
multicast_to_smem[...];  // explain broadcast?


// SPLIT barrier
// Region-based analysis won't work too well.
