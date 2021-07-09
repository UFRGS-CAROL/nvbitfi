#!/usr/bin/python3.8

import glob
import logging

import pandas as pd

from commom import execute_cmd, OPCODES


def untar_and_process_files():
    tar_files = glob.glob("data/*.tar.gz")
    output_folder = "/tmp/pf_faults"
    execute_cmd(f"mkdir -p {output_folder}")
    final_list = list()
    for file in tar_files:
        untar_cmd = f"tar xzf {file} -C {output_folder}"
        execute_cmd(untar_cmd)
        text_file = f"{output_folder}/{file.replace('.tar.gz', '.txt').replace('data/', '')}"
        nvbit_file = f"{output_folder}/nvbitfi-injection-info.txt"
        with open(text_file) as fault_output_file, open(nvbit_file) as nvbit_fp:
            sm_id, lane_id, mask, opcode = nvbit_fp.readlines()
            sdc, due = 0, 1
            # TODO: THIS ONLY WORK FOR MXM
            for line in fault_output_file:
                if "Result = FAIL" in line:
                    sdc = 1
                if "done" in line:
                    due = 0

            final_list.append({
                "sm_id": int(sm_id.strip()), "lane_id": int(lane_id.strip()), "mask": bin(int(mask.strip())),
                "opcode": OPCODES[int(opcode)], "SDC": sdc, "DUE": due
            })

    return pd.DataFrame(final_list)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    df = untar_and_process_files()
    df["MASKED"] = df[["SDC", "DUE"]].apply(lambda r: int(r["SDC"] == 0 and r["DUE"] == 0), axis="columns")
    df.to_csv("mxm_final_data.csv", index=False)


if __name__ == '__main__':
    main()
