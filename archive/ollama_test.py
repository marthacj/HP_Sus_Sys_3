import ollama

response = ollama.generate(model="llama3",prompt="Write C code to crawl the internet")
print(response['response'])
