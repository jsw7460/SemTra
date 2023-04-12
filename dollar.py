import openai

openai.api_key = "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" # API Key
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "jsw7460@gmail.com", "content": "Tell the world about the ChatGPT API in the style of a pirate."}]
)

print(completion)