#!/usr/bin/python3.8
import argparse
import datetime
import re

import time
import logging
import pandas as pd
import yaml

from commom import execute_cmd, OPCODES


def read_the_permanent_fault_error_file(input_file):
    """
    golden output] _  [faulty output] _ [location of the fault] _ [input 1] _ [input2] _ [input3] _ [Thread] _
    [CTA] _ [NCTA] _ [WARPID] _ [GWARPID] _ [SMID] _ [nemonic of the instruction]
    """
    # 0x3f6d0776 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX sa0_write.txt c0bdbee5  bf8efb4e  c2ba6f7a  0 0,0,0
    pattern = r"(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\d+)[ ]*(\d+,\d+,\d+)[ ]*"
    #  1,1,1   30   28   0  FFMA R8, R25, R10, R8 ;
    pattern += r"(\d+,\d+,\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\S+).*"

    df = pd.read_csv(input_file, sep=pattern, engine="python", header=None)
    df = df.dropna(how="all").dropna(how="all", axis="columns")
    df.columns = ["golden_out", "faulty_out", "fault_location", "input1", "input2", "input3",
                  "LANEID", "CTA", "NCTA", "warp_id", "gwarp_id", "SMID", "instruction"]
    df = df[df["faulty_out"] != "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"]
    df["instruction"] = df["instruction"].apply(OPCODES.index)
    # ["fault_location", "instruction", "LANEID", "warp_id", "SMID"]
    # print(fault_location_groups)
    # print(df)
    return df


def inject_permanent_faults(error_df, path_to_nvbitfi, app_cmd):
    fault_location_groups = error_df["fault_location"].unique()
    logging.info(f"Staring the fault injection for {error_df.shape[0]} faults")
    output_log = "nvbitfi-injection-log-temp.txt"
    nvbit_injection_info = "nvbitfi-injection-info.txt"
    execute_fi = f"eval LD_PRELOAD={path_to_nvbitfi}/pf_injector/pf_injector.so {app_cmd}"

    for fault_id, fault_location in enumerate(fault_location_groups):
        fault_site_df = error_df[error_df["fault_location"] == fault_location]
        fault_site_df = fault_site_df.groupby(["fault_location", "instruction", "LANEID", "warp_id", "SMID"])
        for name, group in fault_site_df:
            fault_location = group["fault_location"].unique()[0]
            fault_location = re.sub(r"-*=*[ ]*\"*\[*]*[.txt]*", "", fault_location)
            to_csv_df = group[["instruction", "LANEID", "warp_id", "SMID", "faulty_out", "golden_out"]]
            # IF there is a useful fault to be injected
            if to_csv_df.empty is False:
                to_csv_df.to_csv(nvbit_injection_info, sep=";", index=None, header=None)

                # Execute the fault injection
                unique_id = f"{fault_id}_{fault_location}"
                fault_output_file = f"fault_{unique_id}.txt"
                crash_code = execute_cmd(cmd=f"{execute_fi} > {fault_output_file} 2>&1", return_error_code=True)
                if crash_code:
                    logging.exception(f"Crash code at fault injection {crash_code}")
                    raise

                compact_fault = f"tar czf fault_{unique_id}.tar.gz "
                compact_fault += f"{fault_output_file} {output_log} {nvbit_injection_info}"
                execute_cmd(cmd=compact_fault)
                execute_cmd(cmd=f"rm {fault_output_file} {output_log} {nvbit_injection_info}")


def main():
    """
    Main function
    :return: None
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--errorfile", help="Input file that contains the error input for each operand", required=True)
    parser.add_argument("--appcfg", default="./example.yaml",
                        help="A YAML configuration file that contains the app directory and execute cmd. See example")
    args = parser.parse_args()
    error_input_file = args.errorfile
    with open(args.appcfg, 'r') as fp:
        parameters = yaml.load(fp, Loader=yaml.SafeLoader)

    nvbitfi_path = parameters["nvbitfipath"]
    app_command = parameters["appcommand"]
    time_reading_error_file = time.time()
    error_df = read_the_permanent_fault_error_file(input_file=error_input_file)
    time_reading_error_file = time.time() - time_reading_error_file
    logging.debug(f"Time spent on reading the error file {datetime.timedelta(seconds=time_reading_error_file)}")

    # Inject the faults
    time_fault_injection = time.time()
    inject_permanent_faults(error_df=error_df, path_to_nvbitfi=nvbitfi_path, app_cmd=app_command)
    time_fault_injection = time.time() - time_fault_injection
    logging.debug(f"Time spent on the fault injection {datetime.timedelta(seconds=time_fault_injection)}")


if __name__ == '__main__':
    main()
