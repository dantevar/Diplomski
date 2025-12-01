import os
import requests

HF_TOKEN = os.environ["HF_TOKEN"]
API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Učitaj cijeli kontekst iz response.txt
if os.path.exists("response.txt"):
    with open("response.txt", "r", encoding="utf-8") as f:
        previous_response = f.read()
else:
    previous_response = ""

# Novi upit
user_question = "možeš li mi napisati egzatne algoritme krenući od iscrpne pretrage"

payload = {
    "model": "deepseek-ai/DeepSeek-V3.2-Exp",
    "messages": [
        {"role": "system", "content": "Koristi prethodni odgovor iz response.txt kao kontekst prije nego odgovoriš."},
        {"role": "assistant", "content": previous_response},
        {"role": "user", "content": user_question}
    ]
}

resp = requests.post(API_URL, headers=headers, json=payload)

if resp.status_code == 200:
    answer = resp.json()["choices"][0]["message"]["content"]
    print(answer)

    # Dodaj *na kraj* filea
    with open("response.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + answer)  # razmak da se odgovori ne slijepe
else:
    print(resp.status_code)
    print(resp.json())
