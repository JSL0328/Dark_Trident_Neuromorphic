"""
Convert LArCV ROOT files to .npy format for neuromorphic group.

cosmics_corsika_training_set.root  -> label 0 (cosmic)
cosmics_corsika_test_set.root      -> label 0 (cosmic)
ncpi0_only_training_set.root       -> label 1 (shower)
ncpi0_only_test_set.root           -> label 1 (shower)

Output structure:
  neuromorphic/cosmic+ncpi0_full/
    image_npy/
      event_0.npy, event_1.npy, ...   (each 512x512 image)
    label_npy/
      neuromorphic_labels.npy          (N,) array: 0=cosmic, 1=shower
"""

from larcv import larcv
import numpy as np
import os
from ROOT import TChain


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_DIR   = "/vols/sbn/uboone/darkTridents/data/CNN_training/"
OUTPUT_DIR = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/neuromorphic/cosmic+ncpi0_full/"

FILES = [
    (DATA_DIR + "cosmics_corsika_training_set.root", 0),
    (DATA_DIR + "cosmics_corsika_test_set.root",     0),
    (DATA_DIR + "ncpi0_only_training_set.root",      1),
    (DATA_DIR + "ncpi0_only_test_set.root",          1),
]

PLANE   = 0
ADC_MIN = 10
ADC_MAX = 500


# ──────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────
def load_data(path):
    chain = TChain("image2d_image2d_binary_tree")
    chain.AddFile(path)
    return chain


def get_image(chain, event):
    chain.GetEntry(event)
    cpp_obj = chain.image2d_image2d_binary_branch
    image   = larcv.as_ndarray(cpp_obj.as_vector()[PLANE])
    return image


def apply_adc_threshold(image):
    image = np.where(image < ADC_MIN, 0,       image)
    image = np.where(image > ADC_MAX, ADC_MAX, image)
    return image


def main():
    os.makedirs(OUTPUT_DIR + "image_npy", exist_ok=True)
    os.makedirs(OUTPUT_DIR + "label_npy", exist_ok=True)

    all_labels    = []
    event_counter = 0

    for filepath, label in FILES:
        print(f"\nProcessing: {filepath}  (label={label})")

        chain   = load_data(filepath)
        n_total = chain.GetEntries()
        print(f"Total events: {n_total}  ->  Reading all")

        for i in range(n_total):
            image = get_image(chain, i)
            image = apply_adc_threshold(image)

            npy_path = OUTPUT_DIR + f"image_npy/event_{event_counter}.npy"
            np.save(npy_path, image)

            all_labels.append(label)

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n_total} events done")

            event_counter += 1

    label_array = np.array(all_labels)
    label_path  = OUTPUT_DIR + "label_npy/neuromorphic_labels.npy"
    np.save(label_path, label_array)

    print(f"\nTotal events saved: {event_counter}")
    print(f"Label array shape:  {label_array.shape}")
    print(f"Cosmic  (label=0):  {np.sum(label_array == 0)}")
    print(f"Shower  (label=1):  {np.sum(label_array == 1)}")
    print(f"\nImages saved to: {OUTPUT_DIR}image_npy/")
    print(f"Labels saved to: {label_path}")


if __name__ == "__main__":
    main()
