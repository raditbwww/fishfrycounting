import requests


url = "http://localhost:5000/api/friends"
payload = {
    "name":"john doe",
    "role":"SWE",
    "description":"he works at google",
    "gender":"male"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.json())
