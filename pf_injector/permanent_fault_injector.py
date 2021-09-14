#!/usr/bin/python3.8
import argparse
import datetime
import glob
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
    # Create the mask
    df["golden_out"] = df["golden_out"].apply(lambda x: int(x, base=16))
    df["faulty_out"] = df["faulty_out"].apply(lambda x: int(x, base=16))
    df["instruction"] = df["instruction"].apply(OPCODES.index)
    return df


def inject_permanent_faults(error_df, path_to_nvbitfi, app_cmd, app_name):
    logging.info(f"Staring the fault injection for {error_df.shape[0]} faults")
    nvbit_injection_info = "nvbitfi-injection-info.txt"
    output_logs = [nvbit_injection_info, "nvbitfi-injection-log-temp.txt"]
    execute_fi = f"eval LD_PRELOAD={path_to_nvbitfi}/pf_injector/pf_injector.so {app_cmd}"
    logs_foler = f"logs/{app_name}"
    execute_cmd(f"mkdir -p {logs_foler}")
    execute_cmd(f"rm {logs_foler}/* /var/radiation-benchmarks/log/*.log")

    fault_site_df = error_df.groupby(["instruction", "LANEID", "warp_id", "SMID"])
    for fault_id, (name, group) in enumerate(fault_site_df):
        # new_inj_info.injInstType,injLaneID,warpID,injSMID,faulty_out,golden_out
        to_csv_df = group[["instruction", "LANEID", "warp_id", "SMID", "faulty_out", "golden_out"]]
        # IF there is a useful fault to be injected
        if to_csv_df.empty is False:
            # fault_location = group["fault_location"].unique()
            # assert len(fault_location) == 1, "More than 1 fault location"
            # pf_loc = re.sub(r"-*=*[ ]*\"*\[*]*[.txt]*", "", fault_location[0])
            thread_id = '_'.join(map(str, name[1:]))
            # unique_id = f"{fault_id}_{pf_loc}_{thread_id}"
            unique_id = f"{fault_id}_{thread_id}"
            # Save the nvbit input file
            to_csv_df.to_csv(nvbit_injection_info, sep=";", index=None, header=None)
            # Execute the fault injection
            fault_output_file = f"fault_{unique_id}.txt"
            crash_code = execute_cmd(cmd=f"{execute_fi} > {fault_output_file} 2>&1", return_error_code=True)
            if crash_code:
                logging.exception(f"Crash code at fault injection {crash_code}")
                raise

            # rename these files
            # nvbitfi-injection-info.txt  nvbitfi-injection-log-temp.txt
            rad_log = glob.glob("/var/radiation-benchmarks/log/*.log")[0]
            tmp_logs_names = output_logs + [rad_log]
            for mv_file in tmp_logs_names:
                new_name = unique_id + "_" + mv_file
                execute_cmd(f"{mv_file} {logs_foler}/{new_name}")


def main():
    """
    Main function
    :return: None
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    # parser.add_argument("--errorfile",
    # help="Input file that contains the error input for each operand", required=True)
    parser.add_argument("--appcfg", default="./example.yaml",
                        help="A YAML configuration file that contains the app directory and execute cmd. See example")
    args = parser.parse_args()
    # error_input_file = args.errorfile
    with open(args.appcfg, 'r') as fp:
        parameters = yaml.load(fp, Loader=yaml.SafeLoader)

    nvbitfi_path = parameters["nvbitfipath"]
    app_command = parameters["appcommand"]
    app_name = parameters["appname"]
    time_reading_error_file = time.time()
    ####################################################################################################################
    # TODO: QUICK FIX FOR Lenet
    selected_fault_location = "sa1_DUT_i_fpu_mul_mantissa_b_reg[0]_Q.txt"
    all_txts = glob.glob("/home/fernando/temp/matteo_project/lenet_results/*/*.txt", recursive=True)
    final_list = list()
    for esteban_txt in all_txts:
        txt_df = read_the_permanent_fault_error_file(input_file=esteban_txt)
        final_list.append(txt_df)
    error_df = pd.concat(final_list)
    error_df = error_df[error_df["fault_location"] == selected_fault_location]
    ####################################################################################################################

    time_reading_error_file = time.time() - time_reading_error_file
    logging.debug(f"Time spent on reading the error file {datetime.timedelta(seconds=time_reading_error_file)}")

    # Inject the faults
    time_fault_injection = time.time()
    inject_permanent_faults(error_df=error_df, path_to_nvbitfi=nvbitfi_path, app_cmd=app_command, app_name=app_name)
    time_fault_injection = time.time() - time_fault_injection
    logging.debug(f"Time spent on the fault injection {datetime.timedelta(seconds=time_fault_injection)}")


if __name__ == '__main__':
    main()
