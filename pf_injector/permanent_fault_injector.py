#!/usr/bin/python3.8
import argparse
import datetime
import re
import time
import logging
import pandas as pd
import yaml
from commom import execute_cmd, OPCODES


def inject_permanent_faults(error_df, path_to_nvbitfi, app_cmd, app_name):
    logging.info(f"Staring the fault injection for {error_df.shape[0]} faults")
    nvbit_injection_info = "nvbitfi-injection-info.txt"
    output_logs = [nvbit_injection_info, "nvbitfi-injection-log-temp.txt"]
    execute_fi = f"eval LD_PRELOAD={path_to_nvbitfi}/pf_injector/pf_injector.so {app_cmd}"
    logs_foler = f"logs/{app_name}"
    # Clean before start
    execute_cmd(" ".join(["rm -f", f"{logs_foler}/*"] + output_logs))

    execute_cmd(f"mkdir -p {logs_foler}")
    error_df["opcode_id"] = error_df["instruction"].apply(OPCODES.index)
    fault_site_df = error_df.groupby(["fault_location", "instruction", "LANEID", "SMID"])
    for fault_id, (name, group) in enumerate(fault_site_df):
        # IF there is a useful fault to be injected
        if group.empty is False:
            pf_loc = re.sub(r"-*=*[ ]*\"*\[*]*[.txt]*", "", name[0])
            lane_warp_sm_ids = '_'.join(map(str, name[1:]))
            unique_id = f"{fault_id}_{pf_loc}_{lane_warp_sm_ids}"
            # Save the nvbit input file
            kernel_groups = group.groupby("kernel")
            for kn, kg in kernel_groups:
                # ORDER IN THE NVBITFI
                #             injInstType; injLaneID;warpID;    injSMID; injMask;instructionIndex
                to_save = kg[["opcode_id", "LANEID", "warp_id", "SMID", "faulty_out", "instruction_index"]]
                with open(nvbit_injection_info, mode='a') as fp:
                    fp.write(f"{kn};{kg.shape[0]}\n")
                    to_save.to_csv(fp, sep=";", index=None, header=None)
            # Execute the fault injection
            fault_output_file = f"fault_{unique_id}.txt"
            execute_cmd(cmd=f"{execute_fi} > {fault_output_file} 2>&1")
            # rename these files
            # nvbitfi-injection-info.txt  nvbitfi-injection-log-temp.txt and fault_output_file
            tmp_logs_names = output_logs + [fault_output_file]
            for mv_file in tmp_logs_names:
                new_name = unique_id + "_" + mv_file
                execute_cmd(f"mv {mv_file} {logs_foler}/{new_name}")
            if fault_id > 10: break


def main():
    """ Main function """
    logging.basicConfig(level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S',
                        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--appcfg", default="./example.yaml",
                        help="A YAML configuration file that contains the app directory and execute cmd. See example")
    args = parser.parse_args()
    with open(args.appcfg, 'r') as fp:
        parameters = yaml.load(fp, Loader=yaml.SafeLoader)

    nvbitfi_path = parameters["nvbitfipath"]
    app_command = parameters["appcommand"].strip()
    app_name = parameters["appname"]
    csv_database = parameters["csvdatabase"]
    # reading the file
    time_reading_error_file = time.time()
    error_df = pd.read_csv(csv_database)
    time_reading_error_file = time.time() - time_reading_error_file
    logging.debug(f"Time spent on reading the error file {datetime.timedelta(seconds=time_reading_error_file)}")

    # Inject the faults
    time_fault_injection = time.time()
    inject_permanent_faults(error_df=error_df, path_to_nvbitfi=nvbitfi_path, app_cmd=app_command, app_name=app_name)
    time_fault_injection = time.time() - time_fault_injection
    logging.debug(f"Time spent on the fault injection {datetime.timedelta(seconds=time_fault_injection)}")


if __name__ == '__main__':
    main()
