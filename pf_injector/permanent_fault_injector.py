#!/usr/bin/python3.8
import argparse
import datetime
import time
import logging
import pandas as pd

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

    df["golden_out"] = df["golden_out"].apply(int, base=16)
    df["input1"] = df["input1"].apply(int, base=16)
    df["input2"] = df["input2"].apply(int, base=16)
    df["input3"] = df["input3"].apply(int, base=16)

    print(df.groupby(["fault_location", "instruction", "warp_id", "SMID"]).count().shape)
    print(df.groupby(["fault_location", "instruction", "LANEID", "warp_id", "SMID"]).count().shape)
    print(df)
    return df


def inject_permanent_faults(error_df):
    # logging.info(f"Staring the fault injection for {error_list.shape[0]} faults")
    # output_log = "nvbitfi-injection-log-temp.txt"
    # nvbit_injection_info = "nvbitfi-injection-info.txt"
    # execute_fi = f"eval LD_PRELOAD={path_to_pf_lib}/pf_injector.so {app_cmd}"
    # for fault_id, descriptor in enumerate(error_list):
    #     # Write the fault description
    #     descriptor.write_to_file(nvbit_injection_info)
    #     # Execute the fault injection
    #     fault_output_file = f"fault_{fault_id}.txt"
    #     crash_code = execute_cmd(cmd=f"{execute_fi} > {fault_output_file} 2>&1", return_error_code=True)
    #     if crash_code:
    #         logging.exception(f"Crash code at fault injection {crash_code}")
    #
    #     compact_fault = f"tar czf fault_{fault_id}.tar.gz {fault_output_file} {output_log} {nvbit_injection_info}"
    #     execute_cmd(cmd=compact_fault)
    #     execute_cmd(cmd=f"rm {fault_output_file} {output_log} {nvbit_injection_info}")
    pass


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
    error_df = read_the_permanent_fault_error_file(input_file=error_input_file)
    time_reading_error_file = time.time() - time_reading_error_file
    logging.debug(f"Time spent on reading the error file {datetime.timedelta(seconds=time_reading_error_file)}")

    # Inject the faults
    time_fault_injection = time.time()
    inject_permanent_faults(error_df=error_df)
    time_fault_injection = time.time() - time_fault_injection
    logging.debug(f"Time spent on the fault injection {datetime.timedelta(seconds=time_fault_injection)}")


if __name__ == '__main__':
    main()
