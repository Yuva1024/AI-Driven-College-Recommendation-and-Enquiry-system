from flaskappnew import init_db, app

with app.app_context():
    try:
        init_db()
        print('DB init succeeded')
    except Exception as e:
        print('DB init failed:', e)
