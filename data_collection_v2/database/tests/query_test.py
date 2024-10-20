import requests
import json
base_url = 'http://localhost:6182'


headers = {
    'Content-Type': 'application/json',
}

data = {
    "start": 1600000000000,
    "end": 1700000000000,
    "queries": [
       {
           "aggregator": "avg",
           "metric": "test.measurement",
           "tags": {
              "host": "*"
           }
       }
    ]
        }
response = requests.post(f'{base_url}/api/query', headers=headers, data=json.dumps(data))
print(response, response.url, response.text)