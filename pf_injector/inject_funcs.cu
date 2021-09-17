/*
 * Copyright 2020, NVIDIA CORPORATION.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdint>
#include <cstdio>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "pf_injector.h"
#include "arch.h"

extern "C" __device__ __noinline__ void
inject_error(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int maxRegs) {
    /**
     * EDIT FOR FLEX GRIP INJECTION
     * I only need to identify the fault site
     * "instruction", "LANEID", "warp_id", "SMID"
     * The good thing is that I can control most of the things from the host side
     * That is, the instruction and the instrumentation is for each instruction
     */
    auto inj_info = (inj_info_t *) piinfo;
    uint32_t verbose_device = *((uint32_t *) pverbose_device);
    auto sm_id = get_smid();
    if (sm_id != inj_info->injSMID)
        return; // This is not the selected SM. No need to proceed.

    auto lane_id = get_laneid();
    if (lane_id != inj_info->injLaneID)
        return; // This is not the selected Lane ID. No need to proceed.

    auto warp_id = get_warpid();
    if (warp_id != inj_info->warpID)
        return; // This is not the selected Warp ID
    /**********************************************************************/

    assert(numDestGPRs > 0);
    uint32_t injAfterVal = 0;
    uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
    if (DUMMY) {
        injAfterVal = injBeforeVal;
    } else {
//        injAfterVal = injBeforeVal ^ inj_info->injMask;
        nvbit_write_reg(destGPRNum, int32_t(inj_info->injMask));
    }
    // updating counter/flag to check whether the error was injected
    if (verbose_device)
        printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x\n", destGPRNum, injBeforeVal,
               nvbit_read_reg(destGPRNum), injAfterVal);
    inj_info->errorInjected = true;
    atomicAdd((unsigned long long *) &inj_info->injNumActivations, 1LL);
}
