from typing import Dict

speed_templates = {
	"non_default":
		[
			"Do {video}, {sequential_requirement}, during {param_applied_skill}, the speed {param} is applied.",
			"When undertaking {video}, ensure {sequential_requirement} and adapt the speed {param} for {param_applied_skill}.",
			"While carrying out {video}, comply with {sequential_requirement} and speed modifications {param} to {param_applied_skill}.",
			"{video} is performed with {sequential_requirement} and speed {param} during {param_applied_skill}.",
			"During {param_applied_skill} in {video}, speed {param} is applied while following {sequential_requirement}.",
			"{sequential_requirement} is maintained during {param_applied_skill} in {video} with speed {param} applied.",
			"{video} involves {param_applied_skill} with speed {param} applied and adherence to {sequential_requirement}.",
			"speed {param} is applied while performing {video} with {sequential_requirement} during {param_applied_skill}."
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
			"Do {video}, {sequential_requirement}, wind {param} blows.",
			"Carry out {video}, with {sequential_requirement} as the wind {param} flows.",
			"While carrying out {video}, comply with {sequential_requirement}, but winds {param}.",
			"{video} involves {sequential_requirement} while the wind {param} blows.",
			"During {video}, {sequential_requirement} is maintained as the wind {param} blows.",
			"While performing {video} and following {sequential_requirement}, the wind {param}.",
			"The wind {param} blows during {video}, while adhering to {sequential_requirement}.",
		    "{sequential_requirement} is upheld as the wind {param} blows."
		]
}

weight_templates = {
	"default": ["Imitate these {video} with order requirement {sequential_requirement} with standard object weight"],
	"non_default":
		[
			"Do {video}, under {sequential_requirement}, with {weight} weight {object}.",
		 	"Replicate {video} while following {sequential_requirement} and incorporating {weight} weight {object}.",
			"Emulate {video} with adherence to {sequential_requirement} and integration of {weight} weight {object}.",
			"Mirror {video} by incorporating {weight} weight {object} and following {sequential_requirement}.",
			"Simulate {video} with {sequential_requirement} and include {weight} weight {object}.",
			"Model {video} by following {sequential_requirement} and including {weight} weight {object}.",
			"Mimic {video} by incorporating {weight} weight {object} while following {sequential_requirement}.",
			"Reproduce {video} with {sequential_requirement} and include {weight} weight {object}.",
			"Duplicate {video} while incorporating {weight} weight {object} and following {sequential_requirement}.",
			"Imitate {video} by including {weight} weight {object} and adhering to {sequential_requirement}.",
			"Recreate {video} with {sequential_requirement} while including {weight} weight {object}."
		]
}

template = {
	"speed": speed_templates,
	"wind": wind_templates,
	"weight": weight_templates
}	# type: Dict[str, Dict[str, list[str]]]
