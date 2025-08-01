import pickle
import subprocess

ckpt = "/home/veluga-g3/pg-vla/ckpts/RealUR5Demo_env4_Act_20250716_145335"

result = subprocess.run(['mv', f"{ckpt}/model_meta_info.pkl", f"{ckpt}/model_meta_info_old.pkl"])
print(result)

# 1. pickle ファイルを読み込む
with open(f"{ckpt}/model_meta_info_old.pkl", "rb") as f:
    obj = pickle.load(f)

# 2. 読み込んだオブジェクトを変更する
# 例えば辞書だったらキーに値を追加・更新
obj["action"]["keys"] = ['command_joint_pos']


# 3. 変更したオブジェクトを別ファイル（または上書き）で保存
with open(f"{ckpt}/model_meta_info.pkl", "wb") as f:
    # protocol は互換性の高い HIGHEST_PROTOCOL を使うのがおすすめ
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)