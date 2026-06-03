"""
get_data.py — download the raw NASA battery dataset into ml/data/raw/.

Run:  python ml/get_data.py

We don't commit the 21 MB raw file to git; this script fetches it so anyone can
reproduce the models from scratch. Source: NASA Li-ion Battery Aging dataset
(real measured discharge cycles), mirrored as CSV on GitHub.
"""

import os
import urllib.request

HERE = os.path.dirname(__file__)
RAW_DIR = os.path.join(HERE, "data", "raw")
URL = "https://raw.githubusercontent.com/fmardero/battery_aging/master/discharge.csv"
DEST = os.path.join(RAW_DIR, "nasa_discharge.csv")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(DEST):
        print(f"Already present: {DEST}")
        return
    print(f"Downloading NASA battery data -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Done ({os.path.getsize(DEST) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
