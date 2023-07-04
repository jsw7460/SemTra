import ast


def safe_eval_to_float(parameter: str):
	parameter = ast.literal_eval(parameter)
	for k, v in parameter.items():
		parameter[k] = float(v)
	return parameter
