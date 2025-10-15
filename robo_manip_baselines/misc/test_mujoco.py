#!/usr/bin/env python3
import time, argparse
import mujoco
import mujoco.viewer as viewer
import numpy as np

def set_free_body_pose(model, data, body_name, pos=(0,0,0.5), quat=(1,0,0,0), linvel=(0.1,0,0), angvel=(0,0,0)):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        print(f"[WARN] body '{body_name}' not found; skip pose set")
        return False

    # body -> first joint id (if any)
    jadr = model.body_jntadr[bid]
    jnum = model.body_jntnum[bid]
    if jnum == 0:
        print(f"[INFO] body '{body_name}' has no joint (fixed). Using XML pos as-is.")
        return False

    jid = jadr  # 最初のジョイントID
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        print(f"[INFO] body '{body_name}' joint is not freejoint. Using XML pos as-is.")
        return False

    qadr = model.jnt_qposadr[jid]   # 7要素: quat(4) + pos(3)
    dadr = model.jnt_dofadr[jid]    # 6要素: angvel(3) + linvel(3)

    # 姿勢設定
    data.qpos[qadr:qadr+4] = quat
    data.qpos[qadr+4:qadr+7] = pos
    # 速度設定
    data.qvel[dadr:dadr+3] = angvel
    data.qvel[dadr+3:dadr+6] = linvel
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    # ycb が freejoint なら初期位置/速度を与える
    set_free_body_pose(m, d, "ycb", pos=(0,0,0.5), linvel=(0.1,0,0))

    with viewer.launch_passive(m, d) as v:
        v.cam.distance = 2.0
        v.cam.azimuth = 145.0
        v.cam.elevation = -20.0
        t0 = time.time()
        while v.is_running() and time.time() - t0 < 20.0:
            mujoco.mj_step(m, d)
            v.sync()

if __name__ == "__main__":
    main()
