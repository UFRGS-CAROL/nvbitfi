# set parameters for fi
# it is an easier way to set all parameters for SASSIFI, it is the same as setting it on specific_param.py

from os import environ

benchmark = environ['BENCHMARK']
NVBITFI_HOME = environ['NVBITFI_HOME']
THRESHOLD_JOBS = int(environ['FAULTS'])

all_apps = {
    'simple_add': [
            NVBITFI_HOME + '/test-apps/simple_add', # workload directory
            'simple_add', # binary name
            NVBITFI_HOME + '/test-apps/simple_add/', # path to the binary file
            1, # expected runtime
            "" # additional parameters to the run.sh
        ],
        
    'lava_mp': [
            NVBITFI_HOME + '/test-apps/lava_mp', # workload directory
            'lava_mp', # binary name
            NVBITFI_HOME + '/test-apps/lava_mp/', # path to the binary file
            1, # expected runtime
            "" # additional parameters to the run.sh
        ],
        
    'gemm_tensorcores': [
            NVBITFI_HOME + '/test-apps/gemm_tensorcores', # workload directory
            'gemm', # binary name
            NVBITFI_HOME + '/test-apps/gemm_tensorcores/', # path to the binary file
            1, # expected runtime
            "" # additional parameters to the run.sh
        ],
        
   'bfs': [
        NVBITFI_HOME + '/test-apps/bfs', # workload directory
        'cudaBFS', # binary name
        NVBITFI_HOME + '/test-apps/bfs/', # path to the binary file
        1, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    'accl': [
        NVBITFI_HOME + '/test-apps/accl', # workload directory
        'cudaACCL', # binary name
        NVBITFI_HOME + '/test-apps/accl/', # path to the binary file
        1, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    'mergesort': [
        NVBITFI_HOME + '/test-apps/mergesort', # workload directory
        'mergesort', # binary name
        NVBITFI_HOME + '/test-apps/mergesort/', # path to the binary file
        1, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    'quicksort': [
        NVBITFI_HOME + '/test-apps/quicksort', # workload directory
        'quicksort', # binary name
        NVBITFI_HOME + '/test-apps/quicksort/', # path to the binary file
        2, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    'hotspot': [
        NVBITFI_HOME + '/test-apps/hotspot', # workload directory
        'hotspot', # binary name
        NVBITFI_HOME + '/test-apps/hotspot/', # path to the binary file
        2, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    'darknet_v2': [
        NVBITFI_HOME + '/test-apps/darknet_v2', # workload directory
        'darknet_v2', # binary name
        NVBITFI_HOME + '/test-apps/darknet_v2/', # path to the binary file
        5, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    
    'gaussian': [
        NVBITFI_HOME + '/test-apps/gaussian', # workload directory
        'cudaGaussian', # binary name
        NVBITFI_HOME + '/test-apps/gaussian/', # path to the binary file
        1, # expected runtime
        "" # additional parameters to the run.sh
    ],
    
    'lud': [
        NVBITFI_HOME + '/test-apps/lud', # workload directory
        'cudaLUD', # binary name
        NVBITFI_HOME + '/test-apps/lud/', # path to the binary file
        1, # expected runtime
        "" # additional parameters to the run.sh
    ],
}

apps = {benchmark : all_apps[benchmark]}
