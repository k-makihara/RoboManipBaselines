import pickle

# 1. pickle ファイルを読み込む
with open("/home/veluga-g3/pg-vla/ckpts/RealUR5eDemo_20250624_190911_Act_20250708_181758/model_meta_info_old.pkl", "rb") as f:
    obj = pickle.load(f)

# 2. 読み込んだオブジェクトを変更する
# 例えば辞書だったらキーに値を追加・更新
obj["action"]["keys"] = ['command_joint_pos']


# 3. 変更したオブジェクトを別ファイル（または上書き）で保存
with open("/home/veluga-g3/pg-vla/ckpts/RealUR5eDemo_20250624_190911_Act_20250708_181758/model_meta_info.pkl", "wb") as f:
    # protocol は互換性の高い HIGHEST_PROTOCOL を使うのがおすすめ
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)