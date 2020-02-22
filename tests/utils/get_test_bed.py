import pandas as pd
import os

def get_test_bed():
    return pd.read_csv("{cwd}/test.bed".format(
        cwd=os.path.dirname(os.path.abspath(__file__))
    ), sep="\t")