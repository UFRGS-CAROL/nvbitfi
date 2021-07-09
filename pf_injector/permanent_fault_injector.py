#!/usr/bin/python3.8
import argparse
import re


class PermanentFaultDescriptor:
    pass


def read_the_permanent_fault_error_file(input_file):
    """
    The format of a line in the file is like bellow, the new line is only for readability
    [golden output] [faulty output] [location of the fault] [input 1] [input2] [input3] [Thread] [CTA] [NCTA] [WARPID]
    [GWARPID] [SMID] [nemonic of the instruction]
    example:
    0x3f8147ae XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX sa0_write.txt 0x3c23d70a 0x3f800000 0x3c23d70a 0 0,1,0 4,4,1 18 146 8
    FFMA R12, R28, R13, R12 ;
    Regarding the location of the fault: sa0 = Stuck-at 0, sa1 = Stuck-at 1
    """
    pattern = r"(0[xX][0-9a-fA-F]+) (0[xX][0-9a-fA-F]+|[xX]+) (\S+) (0[xX][0-9a-fA-F]+)  (0[xX][0-9a-fA-F]+)  "
    pattern += r"(0[xX][0-9a-fA-F]+)  (\d+)  (\d+),(\d+),(\d+)   (\d+),(\d+),(\d+)   (\d+)   (\d+)   (\d+)  "
    pattern += r"(\S+) (\S+), (\S+), (\S+), (\S+) ;.*"
    with open(input_file) as fp:
        for line in fp:
            m = re.match(pattern=pattern, string=line)
            if m and "XXXX" in line:
                print(m.groups())


def inject_permanent_fault():
    pass


def main():
    """
    Main function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--errorfile", help="Input file that contains the error input for each operand")
    args = parser.parse_args()
    error_input_file = args.errorfile
    read_the_permanent_fault_error_file(input_file=error_input_file)


if __name__ == '__main__':
    main()
