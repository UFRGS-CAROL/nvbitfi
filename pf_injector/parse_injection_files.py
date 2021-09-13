#!/usr/bin/python3.8

import glob
import logging
import re

import pandas as pd
from commom import execute_cmd, OPCODES


def untar_and_process_files():
    tar_files = glob.glob("*.tar.gz")
    output_folder = "/tmp/pf_faults"
    execute_cmd(f"mkdir -p {output_folder}")
    final_list = list()
    for file in tar_files:
        execute_cmd(f"tar xzf {file} -C {output_folder}")
        tar_pattern = r"fault_(\d+)_(\S+)_(\d+)_(\d+)_(\d+)_(\d+).tar.gz"
        m_fault = re.match(tar_pattern, file)
        # fault_19_sa1_op0_in20_4_27_6_0.tar.gz
        fault_id, fault_location, opcode, lane_id, warp_id, sm_id = m_fault.groups()
        log_file = glob.glob(f"{output_folder}/var/radiation-benchmarks/log/*.log")[0]
        with open(log_file) as fp:
            sdc, end = False, False
            for line in fp:
                sdc = "SDC" in line or sdc
                end = "END" in line or end

            final_list.append({"fault_id": fault_id, "fault_location": fault_location, "sm_id": int(sm_id.strip()),
                               "lane_id": int(lane_id.strip()), "warp_id": int(warp_id.strip()),
                               "opcode": OPCODES[int(opcode)], "SDC": int(sdc), "DUE": int(end == 0)})

        execute_cmd(f"rm -rf {output_folder}/*")

    return pd.DataFrame(final_list)


def main():
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    df = untar_and_process_files()
    df.to_csv("final_data.csv", index=False)


if __name__ == '__main__':
    main()
