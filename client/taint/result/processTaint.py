import os
import sys
import subprocess
import shlex
import time

def parse(txt_directory):
    result_dict = {}

    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            # Read last line of file
            filepath = os.path.join(txt_directory, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    result_dict[filename] = 0
                    continue
                last_line = lines[-1].strip()
            if "[main] INFO soot.jimple.infoflow.android.SetupApplication - Found" not in last_line:
                num_leaks = 0
            else:
                # Extract number from last line
                num_leaks = int(last_line.split()[-2])
            # Add result to dictionary
            result_dict[filename] = num_leaks
    return result_dict


def diff():
    manual_result_dict = parse("manual-spec-output")
    empty_result_dict = parse("empty-spec-output")
    infer_result_dict = parse("infer-spec-output")

    subjects = set([])
    cnt = 0
    for k in manual_result_dict:
        if k in empty_result_dict and k in infer_result_dict and k in infer_result_dict:
            subjects.add(k)
    for subject in manual_result_dict:
        cnt += 1
        print(subject, empty_result_dict[subject], manual_result_dict[subject], infer_result_dict[subject])
    print(cnt)

if __name__ == "__main__":
    diff()

