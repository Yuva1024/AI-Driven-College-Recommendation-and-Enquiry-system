import requests

base = 'http://127.0.0.1:5000'

print('GET /')
r = requests.get(base + '/')
print(r.status_code, len(r.text))
if 'Find the best engineering colleges' in r.text:
    print('Home page OK')
else:
    print('Home page content not found')

print('\nGET /predict')
r = requests.get(base + '/predict')
print(r.status_code, len(r.text))
if 'TNEA College Predictor' in r.text or 'Predict Colleges' in r.text:
    print('Predict page OK')
else:
    print('Predict page content not found')
