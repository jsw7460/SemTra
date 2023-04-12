import pickle
from tqdm import tqdm
import sys
from pathlib import Path
from typing import Dict

import cv2

sys.path.append("/home/jsw7460/comde/")

from metaworld.policies.sawyer_box_close_v2_policy import SawyerBoxCloseV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_coffee_push_v2_policy import SawyerCoffeePushV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_handle_press_v2_policy import SawyerHandlePressV2Policy
from metaworld.policies.sawyer_lever_pull_v2_policy import SawyerLeverPullV2Policy
from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy
from metaworld.policies.sawyer_plate_slide_side_v2_policy import SawyerPlateSlideSideV2Policy
from meta_world.get_video import SingleTask

if __name__ == '__main__':
	agent_list = {'puck': SawyerPlateSlideSideV2Policy(), 'handle': SawyerHandlePressV2Policy(),
				  'button': SawyerButtonPressV2Policy(), 'coffee': SawyerCoffeePushV2Policy(),
				  'lever': SawyerLeverPullV2Policy(), 'stick': SawyerPegInsertionSideV2Policy(),
				  'box': SawyerBoxCloseV2Policy(), 'drawer': SawyerDrawerCloseV2Policy(),
				  'door': SawyerDoorOpenV2Policy()}

	pos = [['box', 'puck'], ['handle', 'drawer'], ['button', 'lever'], ['door', 'stick']]

	data_prefix = Path("/home/jsw7460/comde_save/eval/")
	date = Path("2023-04-03")
	model = Path("causal_fixed")
	data_suffix = Path("fast")
	data_path = data_prefix / date / model / data_suffix

	with open(data_path, "rb") as f:
		dataset = pickle.load(f)  # type: Dict

	for data in dataset.values():

		video_title = data_prefix / date / model / f"{str(data_suffix)}_video"

		epi_id = 0
		resolution = (224, 224)

		mp4v = cv2.VideoWriter_fourcc(*'mp4v')

		# video_2 = cv2.VideoWriter(f'{vid_title}.mp4', mp4v, 30, (400,400))
		video_2 = cv2.VideoWriter(f'{video_title}.mp4', mp4v, 30, resolution)

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
			img = _env.env.render(resolution=resolution)
			print(img.shape)

		exit()


		for epi_id in range(len(loaded_data)):
			loaded_infos = loaded_data[f'episode_{epi_id}']['infos']
			skill_list = loaded_infos[0]['skill_seq']
			print(infoid)
			env_2 = SingleTask(seed=777, skill_list=skill_list)
			env_2.env.skill_list = skill_list
			env_2.reset()

			done = False
			mode = 0
			from tqdm import tqdm

			for i in tqdm(range(len(loaded_infos))):
				loaded_info = loaded_infos[i]
				mode += loaded_data[f'episode_{epi_id}']['rewards'][i]
				qpos = loaded_info['video_info']['qpos']
				qvel = loaded_info['video_info']['qvel']

				env_2.env.set_state(qpos, qvel)
				img_2 = env_2.render(resolution=resolution)
				BGR_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)

				BGR_2 = cv2.putText(BGR_2, f"epi {epi_id}", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
				# BGR_2 = cv2.putText(BGR_2, " ".join(skill_list) , (0,20), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
				BGR_2 = cv2.putText(BGR_2, str(mode), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
				video_2.write(BGR_2)
				if i > 2000: break
		video_2.release()
		cv2.destroyAllWindows()
