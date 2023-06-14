word_to_num = {
		'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
		'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
		'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
		'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
		'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
		'ninety': 90, 'hundred': 100, 'thousand': 1000
	}

num_to_words = {
		0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
		7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
		13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
		18: 'eighteen', 19: 'nineteen'
	}

# Define word representations for tens multiples
tens_words = {
	2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty',
	7: 'seventy', 8: 'eighty', 9: 'ninety'
}

def number_to_words(num: int):
	# Define word representations for numbers up to 19
	negative = False
	if num < 0:
		negative = True
		num = -num

	if 0 <= num < 20:
		ret = num_to_words[num]
	elif 20 <= num < 100:
		tens = num // 10
		remainder = num % 10
		if remainder == 0:
			ret = tens_words[tens]
		else:
			ret = tens_words[tens] + ' ' + num_to_words[remainder]
	elif 100 <= num < 1000:
		hundreds = num // 100
		remainder = num % 100
		if remainder == 0:
			ret = num_to_words[hundreds] + ' hundred'
		else:
			ret = num_to_words[hundreds] + ' hundred ' + number_to_words(remainder)
	else:
		ret = 'Number out of range'

	# ret = ret.replace(" ", "_")
	if negative:
		ret = "minus " + ret
	return ret


def word_to_number(word_string):

	words = word_string.split('_')
	if words[0] == 'minus':
		is_negative = True
		words = words[1:]
	else:
		is_negative = False

	# Handle single-word numbers
	if len(words) == 1:
		value = word_to_num.get(words[0])
		if value is None:
			return None  # Invalid word found
		return -value if is_negative else value

	total_value = 0
	current_value = 0

	for word in words:
		if word == 'and':
			continue
		value = word_to_num.get(word)
		if value is None:
			return None  # Invalid word found

		if value >= 100:
			if current_value == 0:
				current_value = value
			else:
				current_value *= value
				total_value += current_value
				current_value = 0
		else:
			current_value += value

	total_value += current_value
	return -total_value if is_negative else total_value
