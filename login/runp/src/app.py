from flask import Flask, request, jsonify, session
from auth.login import LoginHandler

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Initialize the login handler
login_handler = LoginHandler()

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    
    if login_handler.validate_credentials(username, password):
        session['user'] = username
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({"message": "Logout successful"}), 200

@app.route('/protected', methods=['GET'])
def protected():
    if 'user' in session:
        return jsonify({"message": f"Welcome {session['user']}!"}), 200
    else:
        return jsonify({"message": "Unauthorized"}), 401

if __name__ == '__main__':
    app.run(debug=True)