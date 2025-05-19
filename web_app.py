from flask import Flask, render_template, Response, jsonify, request, url_for, flash, redirect, session
import app
import os
import logging
import cv2 as cv
from datetime import datetime, timedelta
from functools import wraps
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from logging.handlers import RotatingFileHandler
# ...existing imports...

# Initialize Flask application
app_flask = Flask(__name__, static_folder='static')
app_flask.secret_key = 'your_secret_key'

def setup_logging():
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'app.log')
        
        # Create a logger instance
        logger = logging.getLogger('ASLApp')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Configure file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8',
            delay=False
        )
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set handler levels
        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Force flush on write
        file_handler.flush()
        
        # Test log message
        logger.info('Logging system initialized')
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise

# Initialize logger
logger = setup_logging()

# User database (replace with proper database in production)
user_db = {'admin': 'password123'}

# Global variables
cap = None
latest_landmark = None


def setup_logging():
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'app.log')
        
        # Create a logger instance
        logger = logging.getLogger('ASLApp')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Configure file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8',
            delay=False
        )
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set handler levels
        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Force flush on write
        file_handler.flush()
        
        # Test log message
        logger.info('Logging system initialized')
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise


# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication check for all routes
@app_flask.before_request
def check_auth():
    public_routes = ['login', 'static']
    if (request.endpoint not in public_routes and 
        'username' not in session and 
        request.endpoint is not None):
        return redirect(url_for('login'))

# Route handlers
@app_flask.route('/')
def index():
    if 'username' not in session:
        logger.info('Unauthorized access attempt to index')
        return redirect(url_for('login'))
    logger.info(f'User {session["username"]} accessed index page')
    return render_template('index.html')

@app_flask.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        logger.info(f'Login attempt for user: {username}')
        if username in user_db and user_db[username] == password:
            session['username'] = username
            logger.info(f'Successful login for user: {username}')
            return redirect(url_for('index'))
        logger.warning(f'Failed login attempt for user: {username}')
        flash('Invalid username or password')
    return render_template('login.html')
@app_flask.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app_flask.route('/logout')
def logout():
    username = session.get('username')
    session.clear()
    logger.info(f'User {username} logged out')
    flash('Logged out successfully.')
    return redirect(url_for('login'))

# Video routes
@app_flask.route('/video_feed')
@login_required
def video_feed():
    return Response(app.generate_hand_keypoint_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app_flask.route('/live_video')
@login_required
def live_video():
    return render_template('live_video.html')

@app_flask.route('/sign_language')
@login_required
def sign_language():
    return render_template('sign_language.html')

@app_flask.route('/stop_camera', methods=['POST'])
@login_required
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({'status': 'success'})

@app_flask.route('/latest_landmark')
@login_required
def get_latest_landmark():
    return jsonify({"landmark": app.latest_landmark})

# File upload and model training
@app_flask.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            try:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       'model', 'keypoint_classifier')
                os.makedirs(model_dir, exist_ok=True)
                
                file_path = os.path.join(model_dir, 'keypoint.csv')
                file.save(file_path)
                
                train_model(file_path, model_dir)
                flash('Model training completed successfully!')
                logger.info('Model training completed successfully')
                
            except Exception as e:
                flash(f'Error during file processing: {str(e)}')
                logger.error(f'Error during file processing: {str(e)}')
                
            return redirect(url_for('upload'))
                
        flash('Please upload a valid CSV file')
        return redirect(url_for('upload'))
        
    return render_template('upload.html')

# Label management
@app_flask.route('/add_label', methods=['POST'])
@login_required
def add_label():
    label = request.form.get('label') or request.json.get('label')
    if not label:
        return jsonify({'success': False, 'message': 'No label provided.'}), 400

    label_file = os.path.join(
        os.path.dirname(__file__),
        'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'
    )

    try:
        labels = []
        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]

        if label not in labels:
            with open(label_file, 'a', encoding='utf-8') as f:
                f.write(f"{label}\n")
            logger.info(f'Label added successfully: {label}')
            return jsonify({'success': True, 'message': 'Label added successfully.'})
        return jsonify({'success': False, 'message': 'Label already exists.'}), 409

    except Exception as e:
        logger.error(f'Error adding label: {str(e)}')
        return jsonify({'success': False, 'message': str(e)}), 500

@app_flask.route('/remove_label', methods=['POST'])
@login_required
def remove_label():
    label = request.form.get('label') or request.json.get('label')
    if not label:
        return jsonify({'success': False, 'message': 'No label provided.'}), 400

    label_file = os.path.join(
        os.path.dirname(__file__),
        'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'
    )

    try:
        if not os.path.exists(label_file):
            logger.error('Label file not found')
            return jsonify({'success': False, 'message': 'Label file not found.'}), 404

        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]

        new_labels = [l for l in labels if l.lower() != label.lower()]

        if len(new_labels) == len(labels):
            return jsonify({'success': False, 'message': 'Label not found.'}), 404

        with open(label_file, 'w', encoding='utf-8') as f:
            for l in new_labels:
                f.write(f"{l}\n")

        logger.info(f'Label removed successfully: {label}')
        return jsonify({'success': True, 'message': 'Label removed successfully.'})

    except Exception as e:
        logger.error(f'Error removing label: {str(e)}')
        return jsonify({'success': False, 'message': str(e)}), 500

# Logs viewing
    
@app_flask.route('/logs')
@login_required
def view_logs():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    log_path = os.path.join(log_dir, 'app.log')
    
    # Add styles and buttons
    style = """
        <style>
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px;
                background: #f8f9fa;
                margin-bottom: 20px;
            }
            .btn {
                padding: 8px 16px;
                border-radius: 6px;
                text-decoration: none;
                font-size: 1em;
                cursor: pointer;
                border: none;
            }
            .btn-primary {
                background: #0078D4;
                color: white;
            }
            .btn-danger {
                background: #dc3545;
                color: white;
            }
            .pre-logs {
                background: #222;
                color: #eee;
                padding: 1em;
                border-radius: 8px;
                white-space: pre-wrap;
            }
        </style>
        <script>
            function clearLogs() {
                if (confirm('Are you sure you want to clear all logs?')) {
                    fetch('/clear_logs', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            location.reload();
                        } else {
                            alert('Error clearing logs: ' + data.message);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error clearing logs');
                    });
                }
            }
        </script>
    """
    
    header = f"""
        <div class="header">
            <h2>Application Logs (last 5 minutes)</h2>
            <div>
                <button onclick="clearLogs()" class="btn btn-danger">Clear Logs</button>
                <a href="{url_for('index')}" class="btn btn-primary">Home</a>
            </div>
        </div>
    """

    try:
        if not os.path.exists(log_path):
            logger.error('Log file not found')
            return f"{style}{header}<p>No logs found.</p>"

        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        filtered_lines = []

        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    timestamp_str = line.split(' - ')[0]
                    log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if log_time >= cutoff:
                        filtered_lines.append(line)
                except Exception:
                    filtered_lines.append(line)

        log_content = ''.join(filtered_lines)
        return f"{style}{header}<pre class='pre-logs'>{log_content}</pre>"

    except Exception as e:
        logger.error(f'Error reading log file: {str(e)}')
        return f"{style}{header}<p>Error reading logs: {str(e)}</p>"
    
def train_model(file_path, model_dir):
    try:
        dataset = pd.read_csv(file_path, header=None)
        X = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values
        
        num_classes = len(np.unique(y))
        y = tf.keras.utils.to_categorical(y, num_classes)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
        
        model_save_path = os.path.join(model_dir, 'keypoint_classifier.keras')
        model.save(model_save_path)
        logger.info('Model trained and saved successfully')
        
    except Exception as e:
        logger.error(f'Error during model training: {str(e)}')
        raise

if __name__ == '__main__':
    logger.info('Application started')
    port = int(os.environ.get("PORT", 5000))
    app_flask.run(host='0.0.0.0', port=port, debug=True)

    @app_flask.route('/clear_logs', methods=['POST'])
    @login_required
    def clear_logs():
        try:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            log_file = os.path.join(log_dir, 'app.log')
        
            # Clear the log file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')  # Write empty string to clear file
                
            logger.info('Logs cleared by user: {}'.format(session.get('username')))
            return jsonify({'success': True, 'message': 'Logs cleared successfully'})
        except Exception as e:
            logger.error(f'Error clearing logs: {str(e)}')
            return jsonify({'success': False, 'message': str(e)}), 500