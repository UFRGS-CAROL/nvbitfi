#!/usr/bin/python3.8
import argparse
import datetime
import re
import time
import logging
import pandas as pd
import yaml
from commom import execute_cmd


def inject_permanent_faults(error_df, path_to_nvbitfi, app_cmd, app_name):
    logging.info(f"Staring the fault injection for {error_df.shape[0]} faults")
    nvbit_injection_info = "nvbitfi-injection-info.txt"
    output_logs = [nvbit_injection_info, "nvbitfi-injection-log-temp.txt"]
    execute_fi = f"eval LD_PRELOAD={path_to_nvbitfi}/pf_injector/pf_injector.so {app_cmd}"
    logs_foler = f"logs/{app_name}"
    execute_cmd(f"mkdir -p {logs_foler}")

    fault_site_df = error_df.groupby(["fault_location", "instruction", "LANEID", "warp_id", "SMID"])
    for fault_id, (name, group) in enumerate(fault_site_df):
        # new_inj_info.injInstType,injLaneID,warpID,injSMID,faulty_out,golden_out
        # IF there is a useful fault to be injected
        if group.empty is False:
            pf_loc = re.sub(r"-*=*[ ]*\"*\[*]*[.txt]*", "", name[0])
            thread_id = '_'.join(map(str, name[1:]))
            # unique_id = f"{fault_id}_{pf_loc}_{thread_id}"
            unique_id = f"{fault_id}_{thread_id}"
            # Save the nvbit input file
            group.to_csv(nvbit_injection_info, sep=";", index=None, header=None)
            # Execute the fault injection
            fault_output_file = f"fault_{unique_id}.txt"
            crash_code = execute_cmd(cmd=f"{execute_fi} > {fault_output_file} 2>&1", return_error_code=True)
            assert crash_code != 0, logging.exception(f"Crash code at fault injection {crash_code}")
            # rename these files
            # nvbitfi-injection-info.txt  nvbitfi-injection-log-temp.txt and fault_output_file
            tmp_logs_names = output_logs + [fault_output_file]
            for mv_file in tmp_logs_names:
                new_name = unique_id + "_" + mv_file
                execute_cmd(f"mv {mv_file} {logs_foler}/{new_name}")


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
