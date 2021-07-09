#!/usr/bin/python3.8
import argparse
import datetime
import os
import re
import numpy as np
import time
import logging
from inspect import getframeinfo, stack

OPCODES = ['FADD', 'FADD32I', 'FCHK', 'FCMP', 'FFMA', 'FFMA32I', 'FMNMX', 'FMUL', 'FMUL32I', 'FSEL', 'FSET',
           'FSETP', 'FSWZADD', 'IPA', 'MUFU', 'RRO', 'DADD', 'DFMA', 'DMNMX', 'DMUL', 'DSET', 'DSETP', 'HADD2',
           'HADD2_32I', 'HFMA2', 'HFMA2_32I', 'HMUL2', 'HMUL2_32I', 'HSET2', 'HSETP2', 'IDP', 'IDP4A', 'BFE',
           'BFI', 'BMSK', 'BREV', 'FLO', 'IADD', 'IADD3', 'IADD32I', 'ICMP', 'IMAD', 'IMAD32I', 'IMADSP',
           'IMNMX', 'IMUL', 'IMUL32I', 'ISCADD', 'ISCADD32I', 'ISET', 'ISETP', 'LEA', 'LOP', 'LOP3', 'LOP32I',
           'PLOP3', 'POPC', 'SHF', 'SHL', 'SHR', 'XMAD', 'IMMA', 'HMMA', 'VABSDIFF', 'VADD', 'VMAD', 'VMNMX',
           'VSET', 'VSETP', 'VSHL', 'VSHR', 'VABSDIFF4', 'F2F', 'F2I', 'I2F', 'I2I', 'I2IP', 'FRND', 'MOV',
           'MOV32I', 'PRMT', 'SEL', 'SGXT', 'SHFL', 'CSET', 'CSETP', 'PSET', 'PSETP', 'P2R', 'R2P', 'TEX',
           'TLD', 'TLD4', 'TMML', 'TXA', 'TXD', 'TXQ', 'TEXS', 'TLD4S', 'TLDS', 'STP', 'LD', 'LDC', 'LDG',
           'LDL', 'LDS', 'ST', 'STG', 'STL', 'STS', 'MATCH', 'QSPC', 'ATOM', 'ATOMS', 'RED', 'CCTL', 'CCTLL',
           'ERRBAR', 'MEMBAR', 'CCTLT', 'SUATOM', 'SULD', 'SURED', 'SUST', 'BRA', 'BRX', 'JMP', 'JMX', 'SSY',
           'SYNC', 'CAL', 'JCAL', 'PRET', 'RET', 'BRK', 'PBK', 'CONT', 'PCNT', 'EXIT', 'PEXIT', 'LONGJMP',
           'PLONGJMP', 'KIL', 'BSSY', 'BSYNC', 'BREAK', 'BMOV', 'BPT', 'IDE', 'RAM', 'RTT', 'SAM', 'RPCMOV',
           'WARPSYNC', 'YIELD', 'NANOSLEEP', 'NOP', 'CS2R', 'S2R', 'LEPC', 'B2R', 'BAR', 'R2B', 'VOTE',
           'DEPBAR', 'GETCRSPTR', 'GETLMEMBASE', 'SETCRSPTR', 'SETLMEMBASE', 'PMTRIG', 'SETCTAID']


def execute_cmd(cmd):
    logging.debug(f"Executing {cmd}")
    ret = os.system(cmd)
    caller = getframeinfo(stack()[1][0])
    if ret != 0:
        logging.error(f"ERROR AT: {caller.filename}:{caller.lineno} CMD: {cmd}")
        logging.error(f"Command was not correctly executed error code {ret}")
        raise ValueError()


class PermanentFaultDescriptor:
    def __init__(self, gold_value, fault_value, cta, warp_id, sm_id, instruction):
        self.gold_value = int(gold_value, 16)
        try:
            self.fault_value = int(fault_value, 16)
        except ValueError:
            # When the mask be calculated it will return at FFFFFF
            self.fault_value = int("0xFFFFFFFF", 16) ^ self.gold_value
        self.cta = cta
        self.warp_id = warp_id
        self.sm_id = sm_id
        self.opcode = OPCODES.index(instruction.strip())
        # TODO: FOR THE MXM CASE THE LANE ID IS ALWAYS 0
        self.lane_id = "0"

    @property
    def mask(self):
        return self.gold_value ^ self.fault_value

    def write_to_file(self, output):
        """
        SM ID: 0-Max SMs in the GPU being used
        Lane ID: 0-31
        Mask: uint32 number used to inject error into the destination register (corrupted value = Mask XOR original val)
        opcode ID: 0-171 (see enum InstructionType in common/arch.h for the mapping). 171: all opcodes.
        """
        with open(output, "w") as ofp:
            for line in [self.sm_id, self.lane_id, self.mask, self.opcode]:
                ofp.write(f"{line}\n")


def read_the_permanent_fault_error_file(input_file):
    """
    The format of a line in the file is like bellow, the new line is only for readability
    [golden output] [faulty output] [location of the fault] [input 1] [input2] [input3] [Thread] [CTA] [NCTA] [WARPID]
    [GWARPID] [SMID] [nemonic of the instruction]
    example:
    1                   2                           3           4           5           6      7 8,9,10 11,12,13 14 15  16
    0x3f8147ae XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX sa0_write.txt 0x3c23d70a 0x3f800000 0x3c23d70a 0 0,1,0  4,4,1    18 146 8
    FFMA R12, R28, R13, R12 ;
    Regarding the location of the fault: sa0 = Stuck-at 0, sa1 = Stuck-at 1
    """
    pattern = r"(0[xX][0-9a-fA-F]+) (0[xX][0-9a-fA-F]+|[xX]+) (\S+) (0[xX][0-9a-fA-F]+)  (0[xX][0-9a-fA-F]+)  "
    pattern += r"(0[xX][0-9a-fA-F]+)  (\d+)  (\d+),(\d+),(\d+)   (\d+),(\d+),(\d+)   (\d+)   (\d+)   (\d+)  "
    pattern += r"(\S+) .*"
    values_list = list()
    with open(input_file) as fp:
        for line in fp:
            m = re.match(pattern=pattern, string=line)
            if m:
                descriptor = PermanentFaultDescriptor(gold_value=m.group(1), fault_value=m.group(2),
                                                      cta=m.group(8, 9, 10), warp_id=m.group(14),
                                                      sm_id=m.group(16), instruction=m.group(17).strip())
                values_list.append(descriptor)

    return np.array(values_list)


def inject_permanent_faults(error_list, path_to_pf_lib, app_cmd):
    logging.info(f"Staring the fault injection for {error_list.shape[0]} faults")
    output_log = "nvbitfi-injection-log-temp.txt"
    nvbit_injection_info = f"{path_to_pf_lib}/nvbitfi-injection-info.txt"
    execute_fi = f"eval LD_PRELOAD={path_to_pf_lib}/pf_injector.so {app_cmd}"
    for fault_id, descriptor in enumerate(error_list):
        # Write the fault description
        descriptor.write_to_file(nvbit_injection_info)
        # Execute the fault injection
        fault_output_file = f"fault_{fault_id}.txt"
        execute_cmd(cmd=f"{execute_fi} > {fault_output_file} 2>&1")
        output_new_name = output_log.replace(".txt", f"{fault_id}.txt")
        execute_cmd(cmd=f"mv {output_log} {output_new_name}")
        nvbit_injection_info_new_name = nvbit_injection_info.replace(".txt", f"{fault_id}.txt")
        execute_cmd(cmd=f"mv {nvbit_injection_info} {nvbit_injection_info_new_name}")
        if fault_id == 10:
            break


def main():
    """
    Main function
    :return: None
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--errorfile", help="Input file that contains the error input for each operand")
    args = parser.parse_args()
    error_input_file = args.errorfile
    time_reading_error_file = time.time()
    error_list = read_the_permanent_fault_error_file(input_file=error_input_file)
    time_reading_error_file = time.time() - time_reading_error_file
    logging.debug(f"Time spent on reading the error file {datetime.timedelta(seconds=time_reading_error_file)}")

    # Inject the faults
    # TODO: FIx to load from a yaml file
    PATH_TO_PF = "/home/fernando/NVBITFI/nvbit_release/tools/nvbitfi/pf_injector"
    APP_CMD = "/home/fernando/NVIDIA_CUDA-11.3_Samples/0_Simple/matrixMul/matrixMul -wA=128 -wB=128 -hA=128 -hB=128"
    time_fault_injection = time.time()
    inject_permanent_faults(error_list=error_list, path_to_pf_lib=PATH_TO_PF, app_cmd=APP_CMD)
    time_fault_injection = time.time() - time_fault_injection
    logging.debug(f"Time spent on the fault injection {datetime.timedelta(seconds=time_fault_injection)}")


if __name__ == '__main__':
    main()