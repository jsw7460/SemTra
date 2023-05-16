import pickle
import sys
from pathlib import Path
from typing import Dict

import cv2
from tqdm import tqdm

sys.path.append("/home/jsw7460/comde/")

import d4rl

_ = d4rl
from spirl.rl.envs import kitchen

if __name__ == '__main__':

	# ====================================
	# ========= Hyper parameters =========
	# ====================================
	data_prefix = Path("/home/jsw7460/comde_save/eval/")
	date = Path("2023-05-13")
	model = Path("bcz_kitchen_wind")
	data_suffix = Path("eval")
	data_path = data_prefix / date / model / data_suffix
	# ====================================
	# ========= Hyper parameters =========
	# ====================================

	with open(data_path, "rb") as f:
		dataset = pickle.load(f)  # type: Dict

	video_save_prefix = Path("/home/jsw7460/z4_shared") / date / model / Path("visualiation") / Path(str(data_suffix))

	for data in dataset.values():
		env_name = data["env_name"]
		returns = sum(data["rewards"])
		video_save_prefix.mkdir(parents=True, exist_ok=True)
		video_title = video_save_prefix / Path(f"{returns}_{env_name}")

		epi_id = 0
		resolution = (400, 400)

		mp4v = cv2.VideoWriter_fourcc(*'mp4v')

		video = cv2.VideoWriter(f'{video_title}.mp4', mp4v, 30, resolution)

		infos = data["infos"]
		rewards = data["rewards"]

		_env = getattr(kitchen, "Kitchen_mikebohi")
		_env = _env({"task_elements": ("microwave", "kettle", "bottom burner", "hinge cabinet")})
		_env.reset()

		done = False
		mode = 0

		for i, info in tqdm(enumerate(infos)):
			mode += rewards[i]

			qpos = info["obs_dict"]["org_qpos"]
			qvel = info["obs_dict"]["org_qval"]

			_env._env.set_state(qpos=qpos, qvel=qvel)
			img = _env.render(mode="rgb_array")

			bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			bgr = cv2.putText(bgr, f"Kitchen_mode{mode}", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
			video.write(bgr)
			if i > 1000:
				break
		video.release()
		cv2.destroyAllWindows()

		print(f"Save to {video_title}.mp4")
	exit()
