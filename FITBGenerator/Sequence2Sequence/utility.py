"""
Utility Section for Seq2Seq Trainer
"""

import subprocess


def run_command(command):
    """
    Takes a shell command as an input and runs it onto the shell
    prints the output back to the user
    """

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        encoding="utf-8",
        errors="replace",
    )

    while True:
        realtime_output = process.stdout.readline()

        if realtime_output == "" and process.poll() is not None:
            break

        if realtime_output:
            print(realtime_output.strip(), flush=True)
