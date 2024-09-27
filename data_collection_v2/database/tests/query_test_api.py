import requests

response = requests.get(
    'http://localhost:6182/api/query?start=16000000&m=avg:1m-avg:homeNew{room=Kitchen}',
)

print(response.text)