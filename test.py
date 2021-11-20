import requests

files = {"wave": open('./test/buddytest06.wav', 'rb')}

# API 사용. file을 전송
res1 = requests.post('http://52.231.8.193:5000/test5', files=files)

f1 = open('./test.wav', 'wb')
f1.write(res1.content)
f1.close()
