from larcv import larcv
import numpy as np
import os
from ROOT import TChain

DATA_DIR   = "/vols/sbn/uboone/darkTridents/data/CNN_training/"
OUTPUT_DIR = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/neuromorphic/cosmic_ncpi0_signal_1000/"

FILES = [
    (DATA_DIR + "cosmics_corsika_training_set.root", 0),
    (DATA_DIR + "ncpi0_only_training_set.root",      1),
    (DATA_DIR + "dm_signal_only_training_set.root",  2),
]

N_EVENTS = 1000
PLANE    = 0
ADC_MIN  = 10
ADC_MAX  = 500

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
        n_read  = min(N_EVENTS, n_total)
        print(f"Total events: {n_total}  ->  Reading: {n_read}")

        for i in range(n_read):
            image = get_image(chain, i)
            image = apply_adc_threshold(image)
            np.save(OUTPUT_DIR + f"image_npy/event_{event_counter}.npy", image)
            all_labels.append(label)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_read} events done")
            event_counter += 1

    label_array = np.array(all_labels)
    np.save(OUTPUT_DIR + "label_npy/neuromorphic_labels.npy", label_array)

    print(f"\nTotal events saved: {event_counter}")
    print(f"Label array shape:  {label_array.shape}")
    print(f"Cosmic  (label=0):  {np.sum(label_array == 0)}")
    print(f"Shower  (label=1):  {np.sum(label_array == 1)}")
    print(f"Signal  (label=2):  {np.sum(label_array == 2)}")

if __name__ == "__main__":
    main()