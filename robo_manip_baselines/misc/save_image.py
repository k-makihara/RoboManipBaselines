import h5py
import glob
import matplotlib.pyplot as plt

files = sorted(glob.glob("/groups/gaf51379/physical-grounding/datasets/RealUR5Demo_env5/*.hdf5"))

for file in files:
    f = h5py.File(file)
    plt.imshow(f["front_rgb_image"][0])
    fi = file.split("/")
    plt.savefig(f"{fi[-1][:-5]}.png")
