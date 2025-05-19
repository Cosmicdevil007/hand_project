def hash_password(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)

def generate_session_token(user_id):
    import jwt
    import datetime
    secret_key = "your_secret_key"  # Replace with your actual secret key
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    token = jwt.encode({'user_id': user_id, 'exp': expiration}, secret_key, algorithm='HS256')
    return token

def decode_session_token(token):
    import jwt
    secret_key = "your_secret_key"  # Replace with your actual secret key
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None