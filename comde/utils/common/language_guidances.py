from typing import Dict

# speed_templates = {
# 	"non_default":
# 		[
# 			"Follow the prescribed {sequential_requirement} and replicate the actions demonstrated in this {video}, but make adjustments to the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Mimic the actions in this {video} according to the given {sequential_requirement}, but ensure to adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Imitate this {video} in the specified {sequential_requirement} for order, while making necessary modifications to the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Reproduce the actions shown in this {video} by following the {sequential_requirement} for order, and make sure to adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Emulate this {video} by adhering to the {sequential_requirement} for order restriction, and during {param_applied_skill}, make appropriate adjustments to the speed {param} of the robot arm.",
# 			"Copy the actions demonstrated in this {video} while maintaining the specified {sequential_requirement} for order, and during {param_applied_skill}, modify the speed {param} of the robot arm.",
# 			"Perform the same actions as shown in this {video} while adhering to the {sequential_requirement} for order restriction, and adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Execute the actions in this {video} according to the given {sequential_requirement} for order restriction, and during {param_applied_skill}, ensure to adjust the speed {param} of the robot arm.",
# 			"Duplicate the actions displayed in this {video} following the {sequential_requirement} for order, and while performing {param_applied_skill}, adjust the speed {param} of the robot arm accordingly.",
# 			"Mirror the actions presented in this {video} according to the specified {sequential_requirement} for order, and during {param_applied_skill}, make necessary speed {param} adjustments to the robot arm.",
# 			"Replicate the actions in this {video} in the given {sequential_requirement} for order, and during {param_applied_skill}, make appropriate modifications to the speed {param} of the robot arm.",
# 			"Imitate the actions depicted in this {video} while maintaining the prescribed {sequential_requirement} for order, and during {param_applied_skill}, modify the speed {param} of the robot arm as required.",
# 			"Follow the {sequential_requirement} for order restriction and replicate the actions shown in this {video}, but ensure to adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Mimic this {video} by performing the actions in the specified {sequential_requirement} for order, and during {param_applied_skill}, ensure to adjust the speed {param} of the robot arm.",
# 			"Imitate the actions in this {video} while adhering to the prescribed {sequential_requirement} for order, and adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Follow the {sequential_requirement} to imitate the actions shown in this {video}, and make necessary adjustments to the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Emulate the actions presented in this {video} by adhering to the given {sequential_requirement} for order restriction, and during {param_applied_skill}, modify the speed {param} of the robot arm accordingly.",
# 			"Reproduce the actions demonstrated in this {video} following the ordering constraint of {sequential_requirement}, but adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Duplicate the actions from this {video} while adhering to the prescribed {sequential_requirement} for order, and during {param_applied_skill}, modify the speed {param} of the robot arm.",
# 			"Copy the actions demonstrated in this {video} in the specified {sequential_requirement} for order, but make adjustments to the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Replicate the actions shown in this {video} according to the given {sequential_requirement} for order restriction, but ensure to adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Perform the actions presented in this {video} while following the specified {sequential_requirement} for order, and during {param_applied_skill}, modify the speed {param} of the robot arm.",
# 			"Execute the steps in this {video} in the prescribed {sequential_requirement} for order, but make necessary speed {param} adjustments to the robot arm during {param_applied_skill}.",
# 			"Imitate the actions displayed in this {video} while adhering to the {sequential_requirement} for order restriction, but adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Mimic this {video} by following the specified {sequential_requirement} for order, and during {param_applied_skill}, ensure to adjust the speed {param} of the robot arm.",
# 			"Copy the demonstrated actions in this {video} while maintaining the given {sequential_requirement} for order, and during {param_applied_skill}, modify the speed {param} of the robot arm.",
# 			"Perform the actions in this {video} according to the specified {sequential_requirement} for order restriction, but adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Imitate the steps shown in this {video} following the {sequential_requirement} for order, but make necessary adjustments to the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Recreate the actions presented in this {video} in the prescribed {sequential_requirement} for order, but ensure to adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Follow the {sequential_requirement} to replicate the actions shown in this {video}, but adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Mimic this {video} by adhering to the given {sequential_requirement} for order restriction, and during {param_applied_skill}, make necessary adjustments to the speed {param} of the robot arm.",
# 			"Imitate the actions demonstrated in this {video} while maintaining the specified {sequential_requirement} for order, but adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Follow the {sequential_requirement} for order restriction and imitate the actions shown in this {video}, but adjust the speed {param} of the robot arm during {param_applied_skill}.",
# 			"Mimic this {video} in the specified {sequential_requirement} for order, but during {param_applied_skill}, ensure to adjust the speed {param} of the robot arm.",
# 		]
# 	,
# 	"default":
# 		[
# 			"Replicate the actions presented in this {video} following the {sequential_requirement}, maintaining a standard speed.",
# 			"Mimic the {video} by imitating the steps in the specified {sequential_requirement}, with a regular speed.",
# 			"Imitate this {video} in the prescribed {sequential_requirement}, ensuring a normal speed.",
# 			"Follow the {sequential_requirement} and imitate the actions shown in this {video}, maintaining a typical speed.",
# 			"Emulate the actions demonstrated in this {video} according to the given {sequential_requirement}, with a normal speed.",
# 			"Reproduce the {video} by mimicking the actions in the specified {sequential_requirement}, at a standard speed.",
# 			"Copy the actions from this {video} following the provided {sequential_requirement}, using a normal speed.",
# 			"Perform the actions in this {video} in the order specified by the {sequential_requirement}, at a regular speed.",
# 			"Execute the actions in this {video} according to the {sequential_requirement}, with a normal speed.",
# 			"Imitate the actions demonstrated in this {video} following the {sequential_requirement}, maintaining a typical speed."
# 		]
# }

speed_templates = {
	"non_default":
		[
			"Do {video}, {sequential_requirement}, during {param_applied_skill}, adjust speed by {param}.",
			"When undertaking {video}, ensure {sequential_requirement} and adapt the speed {param} for {param_applied_skill}.",
			"While carrying out {video}, comply with {sequential_requirement} and make speed modifications {param} to {param_applied_skill}."
		]
	,
	"default":
		[
			"Replicate the actions presented in this {video} following the {sequential_requirement}, maintaining a standard speed.",
		]
}


wind_templates = {
	"default":
		[
			"None"
		],
	"non_default":
		[
			"Do {video}, {sequential_requirement}, wind blows from {_from} to {_to} by {param}.",
			"Carry out {video}, with {sequential_requirement} as the wind flows from {_from} to {_to} with {param}.",
			"While carrying out {video}, comply with {sequential_requirement}, but winds from {_from} to {_to} with {param}."
		]
}

template = {
	"speed": speed_templates,
	"wind": wind_templates
}	# type: Dict[str, Dict[str, list[str]]]
