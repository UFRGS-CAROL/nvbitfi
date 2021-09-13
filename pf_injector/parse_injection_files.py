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
        untar_cmd = f"tar xzf {file} -C {output_folder}"
        tar_pattern = r"fault_(\d+)_(\S+)_(\d+)_(\d+)_(\d+)_(\d+).tar.gz"
        m_fault = re.match(tar_pattern, file)
        # fault_19_sa1_op0_in20_4_27_6_0.tar.gz
        if m_fault:
            fault_id, fault_location, instruction, lane_id, warp_id, sm_id = m_fault.groups()
            print(fault_id, fault_location, instruction, lane_id, warp_id, sm_id)
        else:
            print(file)
            exit()
        # final_list.append({
        #     "sm_id": int(sm_id.strip()), "lane_id": int(lane_id.strip()), "mask": bin(int(mask.strip())),
        #     "opcode": OPCODES[int(opcode)], "SDC": sdc, "DUE": due
        # })

    return pd.DataFrame(final_list)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    df = untar_and_process_files()
    # df["MASKED"] = df[["SDC", "DUE"]].apply(lambda r: int(r["SDC"] == 0 and r["DUE"] == 0), axis="columns")
    # df.to_csv("mxm_final_data.csv", index=False)


if __name__ == '__main__':
    main()
