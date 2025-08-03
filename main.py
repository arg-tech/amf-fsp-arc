# main.py

from data_src import data
from framenet import data_calling
from ARC import run_final_training


def run():

    # data.data_handling()
    #
    # data_calling()

    run_final_training()


if __name__ == "__main__":
    run()