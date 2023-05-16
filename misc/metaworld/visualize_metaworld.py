import pickle
import sys
from pathlib import Path
from typing import Dict

import cv2
from tqdm import tqdm

sys.path.append("/home/jsw7460/comde/")

from meta_world.get_video import SingleTask

if __name__ == '__main__':

	pos = [['box', 'puck'], ['handle', 'drawer'], ['button', 'lever'], ['door', 'stick']]

	# ====================================
	# ========= Hyper parameters =========
	# ====================================
	data_prefix = Path("/home/jsw7460/comde_save/eval/")
	date = Path("2023-05-13")
	model = Path("comde_mw_speed_bigmlp")
	data_suffix = Path("slow_puck")
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
		resolution = (224, 224)

		mp4v = cv2.VideoWriter_fourcc(*'mp4v')

		# video_2 = cv2.VideoWriter(f'{vid_title}.mp4', mp4v, 30, (400,400))
		video = cv2.VideoWriter(f'{video_title}.mp4', mp4v, 30, resolution)

		infos = data["infos"]
		rewards = data["rewards"]

		task = infos[0]["skill_seq"]
		_env = SingleTask(seed=777, task=task)
		_env.env.skill_list = task
		_env.reset()

		done = False
		mode = 0

		for i, info in tqdm(enumerate(infos)):
			mode += rewards[i]
			qpos = info["video_info"]["qpos"]
			qvel = info["video_info"]["qvel"]
			_env.env.set_state(qpos=qpos, qvel=qvel)
			img = _env.render(resolution=resolution)
			bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			bgr = cv2.putText(bgr, " ".join(task), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
			video.write(bgr)
			if i > 2000:
				break
		video.release()
		cv2.destroyAllWindows()

		print(f"Save to {video_title}.mp4")
	exit()
