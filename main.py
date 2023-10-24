import os
import argparse

from pdfminer.high_level import extract_text
from brary.vss import cls_pooling, mean_pooling, cosine_similarity, l2_distance  

import numpy as np

import torch
from transformers import pipeline, AutoTokenizer, AutoModel

BOLD = "\033[1m"
ITALIC   = "\033[3m"
GREEN = "\033[92m"
ENDC = "\033[0;0m"# "\033[91m"


def print_help(my_cmd=""):
    msg = f"\nmybrary is a tool for searching notes and documents.\n\n"
    msg += f" Available options:\n"
    msg += f"\th help halp      - {ITALIC} print this help text {ENDC} \n"
    msg += f"\tq quit           - {ITALIC} exit the program {ENDC}\n"

    print(msg)

    return 1

def quit(my_cmd=""):

    return 0

CMD_DICT = {"q": quit, \
        "quit": quit, \
        "exit": quit, \
        "help": print_help, \
        "halp": print_help, \
        "h": print_help,\
        }

def command_loop():

    do_continue = 1
    while do_continue:

        my_cmd = input(">")

        if my_cmd not in CMD_DICT.keys():
            print(f"{my_cmd} is not a valid command.")
            my_cmd = input(f"continue? (y/n)")

            if my_cmd.lower() == "y" or my_cmd.lower() == "yes" or my_cmd.lower() == "yep":
                do_continue = 1
            else:
                do_continue = 0

        else:
            do_continue = CMD_DICT[my_cmd](my_cmd)
            last_cmd = my_cmd

if __name__ == "__main__":

    """
    from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    """
    # Welcome splash

    print("")
    print(f"{GREEN} /\\  /\\  \\\\// |D) |D) /_\\ |D) \\\\//")
    print(f"{GREEN}/  \\/  \\  ||  |D) ||\\ | | ||\\  ||")
    print("")

    print("Welcome to mybrary: my personal library. \n")
    print("\tOr your personal library, I guess, since it's you we're talking to.\n ")
    print("  Unless you're me, of course. \n")
    print(f"{ENDC}")
    
    # print instructions and enter command loop
    print_help()
    command_loop()

