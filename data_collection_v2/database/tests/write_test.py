'''
This file is used to test the write.py file for ticktock db
'''
import requests
import os

base_url = 'http://localhost:6182'
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}
write_data = """homeNew,room=LivingRoom temp=21.1,hum=35.9,co=0 1641024000
homeNew,room=Kitchen temp=21.0,hum=35.9,co=0 1641024000
homeNew,room=LivingRoom temp=21.4,hum=35.9,co=0 1641027600
homeNew,room=Kitchen temp=23.0,hum=36.2,co=0 1641027600
homeNew,room=LivingRoom temp=21.8,hum=36.0,co=0 1641031200
homeNew,room=Kitchen temp=22.7,hum=36.1,co=0 1641031200
homeNew,room=LivingRoom temp=22.2,hum=36.0,co=0 1641034800
homeNew,room=Kitchen temp=22.4,hum=36.0,co=0 1641034800
homeNew,room=LivingRoom temp=22.2,hum=35.9,co=0 1641038400
homeNew,room=Kitchen temp=22.5,hum=36.0,co=0 1641038400
homeNew,room=LivingRoom temp=22.4,hum=36.0,co=0 1641042000
homeNew,room=Kitchen temp=22.8,hum=36.5,co=1 1641042000
homeNew,room=LivingRoom temp=22.3,hum=36.1,co=0 1641045600
homeNew,room=Kitchen temp=22.8,hum=36.3,co=1 1641045600
homeNew,room=LivingRoom temp=22.3,hum=36.1,co=1 1641049200
homeNew,room=Kitchen temp=22.7,hum=36.2,co=3 1641049200
homeNew,room=LivingRoom temp=22.4,hum=36.0,co=4 1641052800
homeNew,room=Kitchen temp=22.4,hum=36.0,co=7 1641052800
homeNew,room=LivingRoom temp=22.6,hum=35.9,co=5 1641056400
homeNew,room=Kitchen temp=22.7,hum=36.0,co=9 1641056400
homeNew,room=LivingRoom temp=22.8,hum=36.2,co=9 1641060000
homeNew,room=Kitchen temp=23.3,hum=36.9,co=18 1641060000
homeNew,room=LivingRoom temp=22.5,hum=36.3,co=14 1641063600
homeNew,room=Kitchen temp=23.1,hum=36.6,co=22 1641063600
homeNew,room=LivingRoom temp=22.2,hum=36.4,co=17 1641067200
homeNew,room=Kitchen temp=22.7,hum=36.5,co=26 1641067200"""

response = requests.post(url=f"{base_url}/api/write", headers=headers, data=write_data)
print("finished", response)