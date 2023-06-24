import requests

ride = {"year":2022,
        "month":3
        }

url = "http://192.168.1.64:9696"
predict = requests.post(url, json = ride)
predict = requests.get(url)
print(predict.json())

