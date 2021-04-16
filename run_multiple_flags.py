#!/usr/bin/python3
import os
import re


def execute(cmd):
    print(cmd)
    result = os.system(cmd)
    if result != 0:
        raise ValueError(f"not possible to run: {cmd}")


def main():
    injections = 1
    current_cwd = os.getcwd()
    flags = ["--maxrregcount=16", "-Xptxas --allow-expensive-optimizations=true", "--use_fast_math"]
    benchmarks = {"gemm"}

    for flag in flags:
        for bench in benchmarks:
            for opt in range(1, 3):
                opt_o = f"O{opt}"
                execute(f"make -C test-apps/{bench} clean")
                execute(f'make -C test-apps/{bench} NVCCOPTFLAGS="{flag}" OPT=-{opt_o}')
                execute(f"make -C test-apps/{bench} generate")
                execute(f"make -C test-apps/{bench} test")

                execute(f"./run_injections.sh {bench} {injections}")
                flag_parsed = re.sub("-*=*", "", flag)
                tar_cmd = f"tar czf {flag_parsed}_{opt_o}_{bench}_nvbitfi_{injections}k.tar.gz "
                tar_cmd += "logs_sdcs_* logs /var/radiation-benchmarks/log/"
                execute(tar_cmd)
                execute("rm -rf /var/radiation-benchmarks/log/*.log logs/* *.csv")


if __name__ == '__main__':
    main()
