import pickle
import matplotlib.pyplot as plt
f = pickle.load(open("/home/veluga-g3/pg-vla/ckpts/RealUR5Demo_env2_Act_20250716_134329/model_meta_info_old.pkl","rb"))
plt.imshow(f["image"]["rgb_example"]["hand"]); plt.show()