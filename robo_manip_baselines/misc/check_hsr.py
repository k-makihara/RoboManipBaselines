import h5py

f = h5py.File("/home/deepstation/rfm/RoboManipBaselines/robo_manip_baselines/dataset/MujocoHsrTidyup_20250617_093824_mt/MujocoHsrTidyup_world0_000.rmb/main.rmb.hdf5")


gripper = f["command_gripper_joint_pos"][:].squeeze(-1)

tr_list = []
for i in range(len(gripper)):
    print(gripper[i],f["command_joint_pos"][i][5])
    if gripper[i] == f["command_joint_pos"][i][5]:
        tr_list.append(True)
    else:
        tr_list.append(False)
print(tr_list)