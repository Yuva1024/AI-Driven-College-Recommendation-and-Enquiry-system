import joblib
import traceback

try:
    joblib.load('saved_models/ensemble_model.pkl')
    print('SUCCESS')
except Exception as e:
    with open('err.log', 'w') as f:
        traceback.print_exc(file=f)
