#!/usr/bin/python3.8
import glob
import os
import re
import pandas as pd


def read_the_permanent_fault_error_file_with_index(input_file):
    """
    case of two inputs:
    [golden output]  [faulty output]  [location of the fault]  [input 1]  [input2]
    [Thread] [CTA]  [NCTA]  [WARPID]  [GWARPID]  [SMID]  [LINEID] [INDEX] [nemonic of the instruction]
    case of three inputs:
    [golden output]  [faulty output]  [location of the fault]  [input 1]  [input2] [input3]
    [Thread] [CTA]  [NCTA]  [WARPID]  [GWARPID]  [SMID]  [LINEID] [INDEX] [nemonic of the instruction]
    """
    #        0x80000000 0x80000100 sa1_DUT_i_fpu_mul_sll_166_M1_0_33_I1.txt bf5c8461  0  15
    pattern_2operands = r"(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\d+,\d+,\d+)[ ]*(\d+,\d+,\d+)"
    pattern_2operands += r"[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\S+)[ ]+\[*[P-R]*(\d+).*"
    pattern_3operands = r"(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\S+)[ ]*(\d+,\d+,\d+)[ ]*(\d+,\d+,\d+)"
    pattern_3operands += r"[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\d+)[ ]*(\S+)[ ]+\[*[P-R]*(\d+).*"
    data = list()
    with open(input_file) as fp:
        #     [golden output]  [faulty output]  [location of the fault]  [input 1]  [input2] [Thread] [CTA]  [NCTA]
        columns_2operands = ["golden_out", "faulty_out", "fault_location", "input1", "input2", "LANEID", "CTA", "NCTA",
                             #  [WARPID]  [GWARPID]  [SMID]  [LINEID] [INDEX] [nemonic of the instruction]
                             "warp_id", "gwarp_id", "SMID", "store_thread", "instruction_index", "instruction",
                             "out_reg"]
        columns_3operands = columns_2operands[:]
        columns_3operands.insert(5, "input3")
        for line in fp:
            m = re.match(pattern_2operands, line)
            if m:
                new = {col: m.group(i + 1) for i, col in enumerate(columns_2operands)}
                new["input3"] = None
            else:
                raise NotImplementedError("Terminar")
                # m = re.match(pattern_3operands, line)
                # assert m, f"pau na linha {line}"
                # new = {col: m.group(i + 1) for i, col in enumerate(columns_3operands)}
            data.append(new)

    df = pd.DataFrame(data)
    df = df[~df["faulty_out"].str.contains("XXXXXXXXXXXXXXXXXXXXXXXX")]
    # Create the mask
    df["golden_out"] = df["golden_out"].apply(lambda x: int(x, base=16))
    df["faulty_out"] = df["faulty_out"].apply(lambda x: int(x, base=16))
    return df


def main():
    base_folder = "/home/fernando/temp/matteo_project"
    lenet_results = f"{base_folder}/new_lenet_results/*"
    test_final_csv = f"{base_folder}/lenet_results/final_results_new.csv"
    try:
        os.remove(test_final_csv)
    except OSError:
        print("File does not exist")
    esteban_folders = glob.glob(lenet_results)
    final_list = list()
    for esteban_folder in esteban_folders:
        print(esteban_folder)
        # Load all txt files in the folder into
        layer_txts = glob.glob(f"{esteban_folder}/*.txt")
        layer_df = pd.concat(list(map(read_the_permanent_fault_error_file_with_index, layer_txts)))
        # layer_df['count'] = layer_df.groupby("instruction")['instruction'].transform('count')
        # layer_df.sort_values('count', inplace=True, ascending=False)
        # layer_df.set_index(["fault_location", "instruction", "LANEID", "SMID"], inplace=True)
        # print(layer_df.reset_index())

        kernel_name, _ = re.match(r"(\S+)_(\d+)", os.path.basename(esteban_folder)).groups()
        layer_df["kernel"] = kernel_name
        final_list.append(layer_df)

    columns_list = ["fault_location", "instruction", "kernel", "LANEID", "SMID", "warp_id", "faulty_out",
                    "instruction_index"]
    final_df = pd.concat(final_list)
    print(final_df.reset_index(drop=True))
    df = final_df[columns_list]
    df.to_csv(test_final_csv, index=False)


main()
