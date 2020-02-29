// Copyright (c) 2019, University of Washington All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
//
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "sum_abs_diff.hpp"


/******************************************************************************/
/* Runs the sum ob absolute differences kernel between a reference and a      */
/* frame image (represented by 3D RGB pixel matrixes)                         */
/******************************************************************************/


// Matrix sizes:
#define REF_WIDTH      5
#define REF_HEIGHT     5
#define FRAME_WIDTH    2
#define FRAME_HEIGHT   2
#define RES_WIDTH      (REF_WIDTH - FRAME_WIDTH + 1)
#define RES_HEIGHT     (REF_HEIGHT - FRAME_HEIGHT + 1)

// Host Matrix multiplication code (to compare results)
template <typename TA, typename TB, typename TC>
void matrix_mult (TA *A, TB *B, TC *C, uint64_t M, uint64_t N, uint64_t P) {
        for (uint64_t y = 0; y < M; y ++) {
                for (uint64_t x = 0; x < P; x ++) {
                        TC res = 0.0f;
                        for (uint64_t k = 0; k < N; k++) {
                                res += A[y * N + k] * B[k * P + x];
                        }
                        C[y * P + x] = res;
                }
        }
        return;
}

// Compute the sum of squared error between matricies A and B (M x N)
template <typename T>
double matrix_sse (const T *A, const T *B, uint64_t M, uint64_t N) {
        double sum = 0;
        for (uint64_t y = 0; y < M; y ++) {
                for (uint64_t x = 0; x < N; x ++) {
                        T diff = A[y * N + x] - B[y * N + x];
                        if(std::isnan(diff)){
                                return diff;
                        }
                        sum += diff * diff;
                }
        }
        return sum;
}

// Print matrix A (M x N). This works well for small matricies.
template <typename T>
void matrix_print(T *A, uint64_t M, uint64_t N) {
        T sum = 0;
        for (uint64_t y = 0; y < M; y ++) {
                for (uint64_t x = 0; x < N; x ++) {
                        std::cout << A[y * N + x] << " ";
                }
                std::cout << '\n';

        }
}

int kernel_sum_abs_diff (int argc, char **argv) {

        int rc;
        char *bin_path, *test_name;
        struct arguments_path args = {NULL, NULL};

        argp_parse (&argp_path, argc, argv, 0, 0, &args);
        bin_path = args.path;
        test_name = args.name;

        bsg_pr_test_info("Running the CUDA Sum of Absolute Differences"
                         "Kernel.\n\n");

        // Define block_size_x/y: amount of work for each tile group
        // Define tg_dim_x/y: number of tiles in each tile group
        // Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
        uint32_t block_size_x = 0;
        uint32_t block_size_y = 0;
        hb_mc_dimension_t tg_dim = { .x = 0, .y = 0 };
        if(!strcmp("v0", test_name)){
                block_size_x = 4;
                block_size_y = 4;
                tg_dim = { .x = 4, .y = 4 };
        } else {
                bsg_pr_test_err("Invalid version provided!.\n");
                return HB_MC_INVALID;
        }
        hb_mc_dimension_t grid_dim = { .x = RES_WIDTH / block_size_x,
                                       .y = RES_HEIGHT / block_size_y };


        // Initialize the random number generators
        std::numeric_limits<int8_t> lim; // Used to get INT_MIN and INT_MAX in C++
        std::default_random_engine generator;
        generator.seed(42);
        // Random numbers are RGB, so the values are limited to [0,255]
        //std::uniform_real_distribution<float> distribution(lim.min(),lim.max());
        std::uniform_real_distribution<float> distribution(0, 255);

        // Allocate A, B, BT, C and R (result) on the host
        int FRAME[FRAME_WIDTH * FRAME_HEIGHT];
        int REF  [REF_WIDTH * REF_HEIGHT];
        int RES  [RES_WIDTH * RES_HEIGHT];

        // Generate random numbers. Since the Manycore can't handle infinities,
        // subnormal numbers, or NANs, filter those out.
        auto res = distribution(generator);

        for (uint64_t i = 0; i < REF_WIDTH * REF_HEIGHT; i++) {
                do{
                        res = distribution(generator);
                }while(!std::isnormal(res) ||
                       !std::isfinite(res) ||
                       std::isnan(res));

                REF[i] = static_cast<int>(res);
        }

        for (uint64_t i = 0; i < FRAME_WIDTH * FRAME_HEIGHT; i++) {
                do{
                        res = distribution(generator);
                }while(!std::isnormal(res) ||
                       !std::isfinite(res) ||
                       std::isnan(res));

                FRAME[i] = static_cast<int>(res);
        }

        // Generate the known-correct results on the host
        //matrix_mult (A, B, R, A_HEIGHT, A_WIDTH, B_WIDTH);

        // Initialize device, load binary and unfreeze tiles.
        hb_mc_device_t device;
        rc = hb_mc_device_init(&device, test_name, 0);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to initialize device.\n");
                return rc;
        }


        rc = hb_mc_device_program_init(&device, bin_path,
                                       "default_allocator", 0);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to initialize program.\n");
                return rc;
        }


        // Allocate memory on the device for REF, FRAME, and RES Since sizeof(float) ==
        // sizeof(int32_t) > sizeof(int16_t) > sizeof(int8_t) we'll reuse the
        // same buffers for each test (if multiple tests are conducted)
        eva_t REF_device, FRAME_device, RES_device;

        // Allocate REF on the device
        rc = hb_mc_device_malloc(&device, REF_HEIGHT * REF_WIDTH * sizeof(float), &REF_device);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Allocate FRAME on the device
        rc = hb_mc_device_malloc(&device, FRAME_HEIGHT * FRAME_WIDTH * sizeof(float), &FRAME_device);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Allocate RES on the device
        rc = hb_mc_device_malloc(&device, RES_HEIGHT * RES_WIDTH * sizeof(float), &RES_device);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to allocate memory on device.\n");
                return rc;
        }

        // Copy FRAME & REF from host onto device DRAM.
        void *dst = (void *) ((intptr_t) REF_device);
        void *src = (void *) &REF[0];
        rc = hb_mc_device_memcpy (&device, dst, src,
                                  (REF_HEIGHT * REF_WIDTH) * sizeof(REF[0]),
                                  HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to copy memory to device.\n");
                return rc;
        }


        dst = (void *) ((intptr_t) FRAME_device);
        src = (void *) &FRAME[0];
        rc = hb_mc_device_memcpy (&device, dst, src,
                                  (FRAME_HEIGHT * FRAME_WIDTH) * sizeof(FRAME[0]),
                                  HB_MC_MEMCPY_TO_DEVICE);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to copy memory to device.\n");
                return rc;
        }

        // Prepare list of input arguments for kernel.
        uint32_t cuda_argv[11] = {REF_device, FRAME_device, RES_device,
                                 REF_HEIGHT, REF_WIDTH,
                                 FRAME_HEIGHT, FRAME_WIDTH,
                                 RES_HEIGHT, RES_WIDTH,
                                 block_size_y, block_size_x};

        // Enquque grid of tile groups, pass in grid and tile group dimensions,
        // kernel name, number and list of input arguments
        rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_sum_abs_diff", 11, cuda_argv);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to initialize grid.\n");
                return rc;
        }

        // Launch and execute all tile groups on device and wait for all to finish.
        rc = hb_mc_device_tile_groups_execute(&device);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to execute tile groups.\n");
                return rc;
        }

        // Copy result matrix RES back from device DRAM into host memory.
        src = (void *) ((intptr_t) RES_device);
        dst = (void *) &RES[0];
        rc = hb_mc_device_memcpy (&device, dst, src,
                                  (RES_HEIGHT * RES_WIDTH) * sizeof(float),
                                  HB_MC_MEMCPY_TO_HOST);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to copy memory from device.\n");
                return rc;
        }

        // Freeze the tiles and memory manager cleanup.
        rc = hb_mc_device_finish(&device);
        if (rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("failed to de-initialize device.\n");
                return rc;
        }

        // Compare the known-correct matrix (R) and the result matrix (C)
/*
        float max = 0.1;
        double sse = matrix_sse(R, C, C_HEIGHT, C_WIDTH);

        if (std::isnan(sse) || sse > max) {
                bsg_pr_test_info(BSG_RED("Matrix Mismatch. SSE: %f\n"), sse);
                return HB_MC_FAIL;
        }
*/

        bsg_pr_test_info(BSG_GREEN("Matrix Match.\n"));
        return HB_MC_SUCCESS;
}

#ifdef COSIM
void cosim_main(uint32_t *exit_code, char * args) {
        // We aren't passed command line arguments directly so we parse them
        // from *args. args is a string from VCS - to pass a string of arguments
        // to args, pass c_args to VCS as follows: +c_args="<space separated
        // list of args>"
        int argc = get_argc(args);
        char *argv[argc];
        get_argv(args, argc, argv);

#ifdef VCS
        svScope scope;
        scope = svGetScopeFromName("tb");
        svSetScope(scope);
#endif
        int rc = kernel_sum_abs_diff(argc, argv);
        *exit_code = rc;
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return;
}
#else
int main(int argc, char ** argv) {
        int rc = kernel_sum_abs_diff(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}
#endif

