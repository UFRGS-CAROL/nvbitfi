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
#include <cassert>
#include <pthread.h>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <csignal>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include "globals.h"
#include "pf_injector.h"

/**
 * New functions and variables for FLexGrip Injection
 */
#define FATAL(error) throw std::runtime_error(std::string("ERROR ") + __FILE__ + ":" + std::to_string(__LINE__));

int verbose;
__managed__ int verbose_device;
int limit = INT_MAX;

// injection parameters input filename: This file is created the script
// that launched error injections (MUST BE GLOBAL)
std::string injInputFilename;

pthread_mutex_t mutex;

//__managed__ inj_info_t inj_info;

__managed__ inj_info_t *managed_inj_info_array = nullptr;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;


void reset_inj_info(int i = 0) {
    managed_inj_info_array[i].injInstType = 0;
    managed_inj_info_array[i].injSMID = 0;
    managed_inj_info_array[i].injLaneID = 0;
    managed_inj_info_array[i].injMask = 0;
    managed_inj_info_array[i].injNumActivations = 0;
    managed_inj_info_array[i].errorInjected = false;
    managed_inj_info_array[i].warpID = 0;
}

// for debugging 
void print_inj_info(int i = 0) {
    assert(fout.good());
    std::cout << "InstType=" << managed_inj_info_array[i].injInstType << ", SMID=" << managed_inj_info_array[i].injSMID
              << ", LaneID=" << managed_inj_info_array[i].injLaneID
              << ", WarpID=" << managed_inj_info_array[i].warpID;
    std::cout << ", Mask=" << managed_inj_info_array[i].injMask << std::endl;
}

// Parse error injection site info from a file. This should be done on host side.
void parse_params(const std::string &filename, int i = 0) {
    static bool parse_flag = false; // file will be parsed only once - performance enhancement
    if (!parse_flag) {
        parse_flag = true;
        reset_inj_info();

        std::ifstream ifs(filename, std::ifstream::in);
        if (ifs.is_open()) {
            ifs >> managed_inj_info_array[i].injSMID;
            assert(managed_inj_info_array[i].injSMID < 1000); // we don't have a 1000 SM system yet.

            ifs >> managed_inj_info_array[i].injLaneID;
            assert(managed_inj_info_array[i].injLaneID < 32); // Warp-size is 32 or less today.

            ifs >> managed_inj_info_array[i].injMask;

            ifs >> managed_inj_info_array[i].injInstType; // instruction type
            assert(managed_inj_info_array[i].injInstType <=
                   NUM_ISA_INSTRUCTIONS); // ensure that the value is in the expected range

        } else {
            printf(" File %s does not exist!", filename.c_str());
            printf(" This file should contain enough information about the fault site to perform "
                   "a permanent error injection run: ");
            printf("(1) SM ID, (2) Lane ID (within a warp), (3) 32-bit mask (as int32), "
                   "(4) Instruction type (as integer, see maxwell_pascal.h). \n");
            assert(false);
        }
        ifs.close();

        if (verbose) {
            print_inj_info();
        }
    }
}

// Parse error injection site info from a file. This should be done on host side.
void parse_flex_grip_file(const std::string &filename) {
    std::ifstream input_file(filename);
    std::vector<inj_info_t> host_database_inj_info;
    if (input_file.good()) {
        // Read the file that contains the error model from FlexGrip
        while (!input_file.eof()) {
            std::string line, word;
            std::vector<std::string> row;
            // read an entire row and
            // store it in a string variable 'line'
            std::getline(input_file, line);
            if (input_file.eof()) {
                break;
            }
            // used for breaking words
            std::stringstream s(line);

            // read every column data of a row and
            // store it in a string variable, 'word'
            while (std::getline(s, word, ';')) {
                // add all the column data
                // of a row to a vector
                row.push_back(word);
            }
            //add to the vector the faulty values
            inj_info_t new_inj_info;
            new_inj_info.injInstType = std::stoul(row[0]);
            new_inj_info.injLaneID = std::stoul(row[1]);
            new_inj_info.warpID = std::stoul(row[2]);
            new_inj_info.injSMID = std::stoul(row[3]);
            auto faulty_out = std::stoul(row[4]);
            auto golden_out = std::stoul(row[5]);
            new_inj_info.injMask = faulty_out ^ golden_out;
            host_database_inj_info.push_back(new_inj_info);
        }
        // COPY to gpu the array of injections
        CUDA_SAFECALL(cudaMallocManaged(&managed_inj_info_array,
                                        host_database_inj_info.size() * sizeof(inj_info_t)));
        std::copy(host_database_inj_info.begin(), host_database_inj_info.end(), managed_inj_info_array);
        CUDA_SAFECALL(cudaDeviceSynchronize());
    } else {
        FATAL("Not possible to open the file " + filename)
    }
}

void update_verbose() {
    static bool update_flag = false; // update it only once - performance enhancement
    if (!update_flag) {
        update_flag = true;
        cudaDeviceSynchronize();
        verbose_device = verbose;
        cudaDeviceSynchronize();
    }
}

int get_maxregs(CUfunction func) {
    int maxregs = -1;
    cuFuncGetAttribute(&maxregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
    return maxregs;
}

void INThandler(int sig) {
    signal(sig, SIG_IGN); // disable Ctrl-C

    fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;";
    fout << " injNumActivations: " << managed_inj_info_array[0].injNumActivations << ":::";
    fout.flush();
    exit(-1);
}


/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
// DO NOT USE UVM (__managed__) variables in this function
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    injInputFilename = "nvbitfi-injection-info.txt";
    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    if (getenv("TOOL_VERBOSE")) {
        verbose = std::stoi(getenv("TOOL_VERBOSE"));
    } else {
        verbose = 0;
    }

    if (getenv("INPUT_INJECTION_INFO")) {
        injInputFilename = getenv("INPUT_INJECTION_INFO");
    }
    if (getenv("OUTPUT_INJECTION_LOG")) {
        injOutputFilename = getenv("OUTPUT_INJECTION_LOG");
    }
    if (getenv("INSTRUMENTATION_LIMIT")) {
        limit = std::stoi(getenv("INSTRUMENTATION_LIMIT"));
    }

    // GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool (1, 2, 3,..)");

    initInstTypeNameMap();

    signal(SIGINT, INThandler); // install Ctrl-C handler

    open_output_file(injOutputFilename);
    if (verbose) printf("nvbit_at_init:end\n");
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

//    parse_params(injInputFilename);  // injParams are updated based on injection seed file
    parse_flex_grip_file(injInputFilename);
    update_verbose();

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f: related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        std::string kname = removeSpaces(nvbit_get_func_name(ctx, f));
        /* Get the vector of instruction composing the loaded CUFunction "func" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        int maxregs = get_maxregs(f);
        assert(fout.good());
        fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "("
             << maxregs << ")" << std::endl;
        for (auto i: instrs) {
            std::string opcode = i->getOpcode();
            std::string instTypeStr = extractInstType(opcode);
            int instType = instTypeNameMap[instTypeStr];
            if (verbose) printf("extracted instType: %s, ", instTypeStr.c_str());
            if (verbose) printf("index of instType: %d\n", instTypeNameMap[instTypeStr]);

            /**
             * MODIFICATION FOR FLEXGRIP PF injection
             */
            if ((uint32_t) instType == managed_inj_info_array[0].injInstType ||
                managed_inj_info_array[0].injInstType == NUM_ISA_INSTRUCTIONS) {
                if (verbose) {
                    printf("instruction selected for instrumentation: ");
                    i->print();
                }

                // Tokenize the instruction
                std::vector<std::string> tokens;
                std::string buf; // a buffer string
                std::stringstream ss(i->getSass()); // Insert the string into a stream
                while (ss >> buf)
                    tokens.push_back(buf);

                int destGPRNum = -1;
                int numDestGPRs = 0;

                if (tokens.size() > 1) { // an actual instruction that writes to either a GPR or PR register
                    if (verbose) printf("num tokens = %ld \n", tokens.size());
                    int start = 1; // first token is opcode string
                    if (tokens[0].find('@') != std::string::npos) { // predicated instruction, ignore first token
                        start = 2; // first token is predicate and 2nd token is opcode
                    }

                    // Parse the first operand - this is the first destination
                    int regnum1 = -1;
                    int regtype = extractRegNo(tokens[start], regnum1);
                    if (regtype == 0) { // GPR reg
                        destGPRNum = regnum1;
                        numDestGPRs = (getOpGroupNum(instType) == G_FP64) ? 2 : 1;

                        int szStr = extractSize(opcode);
                        if (szStr == 128) {
                            numDestGPRs = 4;
                        } else if (szStr == 64) {
                            numDestGPRs = 2;
                        }

                        nvbit_insert_call(i, "inject_error", IPOINT_AFTER);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &managed_inj_info_array[0]);
                        nvbit_add_call_arg_const_val64(i, (uint64_t) &verbose_device);

                        nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
                        if (destGPRNum != -1) {
                            nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
                        } else {
                            nvbit_add_call_arg_const_val32(i, (unsigned int) -1); // destination GPR register val
                        }
                        nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers

                        nvbit_add_call_arg_const_val32(i, maxregs); // max regs used by the inst info
                        /**********************************************************************************************/
                    }
                    // If an instruction has two destination registers, not handled!! (TODO: Fix later)
                }
            }
        }
    }
}

/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch ||
        cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid ||
        cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {

        /* cast params to cuLaunch_params since if we are here we know these are
         * the right parameters type */
        auto *p = (cuLaunch_params *) params;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            if (kernel_id < limit) {
                instrument_function_if_needed(ctx, p->f);
                // cudaDeviceSynchronize();

                nvbit_enable_instrumented(ctx, p->f, true); // run the instrumented version
                // cudaDeviceSynchronize();
            } else {
                nvbit_enable_instrumented(ctx, p->f, false); // do not use the instrumented version
            }

        } else {
            if (kernel_id < limit) {
                if (verbose) printf("is_exit\n");
                cudaDeviceSynchronize();

                cudaError_t le = cudaGetLastError();

                std::string kname = removeSpaces(nvbit_get_func_name(ctx, p->f));
                unsigned num_ctas = 0;
                if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                    cbid == API_CUDA_cuLaunchKernel) {
                    auto *p2 = (cuLaunchKernel_params *) params;
                    num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                }
                assert(fout.good());
                fout << "Injection data; ";
                fout << "index: " << kernel_id << ";";
                fout << "kernel_name: " << kname << ";";
                fout << "ctas: " << num_ctas << ";";
                fout << "selected SM: " << managed_inj_info_array[0].injSMID << ";";
                fout << "selected Lane: " << managed_inj_info_array[0].injLaneID << ";";
                fout << "selected Warp: " << managed_inj_info_array[0].warpID << ";";
                fout << "selected Mask: " << managed_inj_info_array[0].injMask << ";";
                fout << "selected InstType: " << managed_inj_info_array[0].injInstType << ";";
                fout << "injNumActivations: " << managed_inj_info_array[0].injNumActivations << std::endl;

                if (cudaSuccess != le) {
                    assert(fout.good());
                    fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); " << std::endl;
                    exit(1); // let's exit early
                }

                if (verbose) printf("\n index: %d; kernel_name: %s; \n", kernel_id, kname.c_str());
                kernel_id++; // always increment kernel_id on kernel exit

                // cudaDeviceSynchronize();
                pthread_mutex_unlock(&mutex);
            }
        }
    }
}

void nvbit_at_term() {} //nothing to do here
