#!/usr/bin/python3.8
import glob
import logging
import os
import re

import numpy as np
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
                sdc, end = "SDC" in line or sdc, "END" in line or end

            final_list.append({"fault_id": fault_id, "fault_location": fault_location, "sm_id": int(sm_id.strip()),
                               "lane_id": int(lane_id.strip()), "warp_id": int(warp_id.strip()),
                               "opcode": OPCODES[int(opcode)], "SDC": int(sdc), "DUE": int(end == 0)})

        execute_cmd(f"rm -rf {output_folder}/*")

    return pd.DataFrame(final_list)


def extract_output(output):
    numbers = {"Zero": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7, "Eight": 8,
               "Nine": 9}
    with open(output) as fp:
        prob_list, numb_list = [], []
        for line in fp:
            if "Predicted" in line:
                break
        for line in fp:
            m = re.match(r"(.*)%:[ ]*(\S+)", line.strip())
            if m:
                prob_list.append(float(m.group(1).strip()))
                numb_list.append(numbers[m.group(2)])

    return np.array(prob_list), np.array(numb_list)


def parse_lenet_output():
    tar_files = glob.glob("logs/lenet/*.tar.gz")
    tmp_dir = "/tmp/sw_pf"
    prob_gold, numb_gold = extract_output("logs/gold_output.txt")
    execute_cmd(f"mkdir -p {tmp_dir}")

    output_folder = tmp_dir + "/tmptxts"
    execute_cmd(f"mkdir -p {output_folder}")
    fault_list = list()
    for file in tar_files:
        execute_cmd(f"tar xzf {file} -C {output_folder}/")
        #               fault_0_sa0_DUT_U2086_B_FADD_0_0.txt
        tar_pattern = r"(\d+)_(\S+)_(\S+)_(\d+)_(\d+).tar.gz"
        m_fault = re.match(tar_pattern, file)
        # fault_19_sa1_op0_in20_4_27_6_0.tar.gz
        fault_id, fault_location, opcode, lane_id, sm_id = m_fault.groups()
        # IF there is a useful fault to be injected
        pf_loc = re.sub(r"-*=*[ ]*\"*\[*]*[.txt]*", "", fault_location)
        lane_sm_ids = '_'.join([lane_id, sm_id])
        unique_id = f"{fault_id}_{pf_loc}_{lane_sm_ids}"
        # Save the nvbit input file
        fault_output_file = f"{output_folder}/fault_{unique_id}.txt"
        prob_list, numb_list = extract_output(fault_output_file)
        # diffs
        diff_prob = np.sum(np.abs(prob_list - prob_gold))

        # if diff_prob > 1:
        #     print(diff_prob)
        #     if numb_gold[0] != numb_list[0]:
        #         print(numb_gold[0], numb_list[0])
        fault_list.append({
            'SDC': int(diff_prob > 1e-3),
            'Critical SDC': int(numb_gold[0] != numb_list[0]),
            'diff_probability': diff_prob, 'fault_id': fault_id, 'fault_location': pf_loc, 'instruction': opcode,
            'lane_id': lane_id, 'sm_id': sm_id
        })
        execute_cmd(f"rm {output_folder}/*")

    return pd.DataFrame(fault_list)


def main():
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    # df = untar_and_process_files()
    df = parse_lenet_output()
    df.to_csv("final_data.csv", index=False)


if __name__ == '__main__':
    main()
