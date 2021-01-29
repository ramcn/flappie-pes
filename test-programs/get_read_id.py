import os
from ont_fast5_api.fast5_interface import get_fast5_file

def print_all_raw_data():

  for fast5_filepath in os.listdir('./'):
      if fast5_filepath.endswith(".fast5"): 
        with get_fast5_file(fast5_filepath, mode="r") as f5:
            for read in f5.get_reads():
                raw_data = read.get_raw_data()
                print(read.read_id, fast5_filepath,sep='\t')

print_all_raw_data()
