from flask import Flask, render_template, Response, jsonify, request, url_for, flash, redirect
import app  # This imports your app.py
import os
import logging
from datetime import datetime, timedelta
# At the top with other imports
import cv2 as cv

# Initialize Flask application
app_flask = Flask(__name__)
app_flask = Flask(__name__, static_folder='static')
app_flask.secret_key = 'your_secret_key'  # Needed for flashing messages


# Global variable for camera
cap = None

@app_flask.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({'status': 'success'})

def gen_frames():
    global cap
    if cap is None:
        cap = cv.VideoCapture(0)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except:
        if cap is not None:
            cap.release()
            cap = None
        raise

import subprocess

@app_flask.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            # Always save as keypoint.csv in the correct model folder
            save_path = os.path.join('model', 'keypoint_classifier', 'keypoint.csv')
            file.save(save_path)
            flash('File uploaded successfully! Training started...')
            try:
                subprocess.run([
                    "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute", "--inplace",
                    "keypoint_classification.ipynb"
                ], check=True)
                flash('Model training completed!')
            except subprocess.CalledProcessError:
                flash('Model training failed. Please check the notebook for errors.')
        else:
            flash('Please upload a valid CSV file.')
        return redirect(url_for('upload'))
    return render_template('upload.html')
    logging.info('File uploaded successfully!')
logging.error('Model training failed.')



@app_flask.route('/')
def index():
    return render_template('index.html')

@app_flask.route('/video_feed')
def video_feed():
    return Response(app.generate_hand_keypoint_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

#@app_flask.route('/video_feed')
#def video_feed():
 #   return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app_flask.route('/sign_language')
def sign_language():
    return render_template('sign_language.html')

@app_flask.route('/latest_landmark')
def latest_landmark():
    return jsonify({"landmark": app.latest_landmark})

@app_flask.route('/add_label', methods=['POST'])
def add_label():
    # Try to get label from JSON or form
    label = request.form.get('label')
    if not label and request.is_json:
        label = request.json.get('label')
    if not label:
        return jsonify({'success': False, 'message': 'No label provided.'}), 400

    # Absolute path to the label file
    label_file = os.path.join(
        os.path.dirname(__file__),
        'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'
    )

    # Read existing labels
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        labels = []

    # Only add if not already present
    if label not in labels:
        with open(label_file, 'a', encoding='utf-8') as f:
            f.write(label + '\n')
        return jsonify({'success': True, 'message': 'Label added.'})
    else:
        return jsonify({'success': False, 'message': 'Label already exists.'}), 409
    
@app_flask.route('/logs')
def view_logs():
    from datetime import datetime, timedelta
    log_path = 'app.log'
    # Move Home button to the top right
    home_link = '<a href="{}" style="position:fixed;top:16px;right:16px;background:#0078D4;color:#fff;padding:8px 16px;border-radius:6px;text-decoration:none;font-size:1em;">Home</a>'.format(url_for('index'))
    if os.path.exists(log_path):
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        filtered_lines = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming log format: 'YYYY-MM-DD HH:MM:SS,ms LEVEL: message'
                try:
                    timestamp_str = line.split(' ')[0] + ' ' + line.split(' ')[1].split(',')[0]
                    log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    if log_time >= cutoff:
                        filtered_lines.append(line)
                except Exception:
                    # If parsing fails, keep the line (or skip as you wish)
                    filtered_lines.append(line)
        log_content = ''.join(filtered_lines)
        return f"""{home_link}
        <h2>Application Logs (last 5 minutes)</h2>
        <pre style='background:#222;color:#eee;padding:1em;'>{log_content}</pre>"""
    else:
        return f"""{home_link}<h2>No logs found.</h2>"""
    
@app_flask.route('/remove_label', methods=['POST'])
def remove_label():
    # Get label from JSON or form
    label = request.form.get('label')
    if not label and request.is_json:
        label = request.json.get('label')
    if not label:
        return jsonify({'success': False, 'message': 'No label provided.'}), 400

    label_file = os.path.join(
        os.path.dirname(__file__),
        'model', 'keypoint_classifier', 'keypoint_classifier_label.csv'
    )

    if not os.path.exists(label_file):
        return jsonify({'success': False, 'message': 'Label file does not exist.'}), 404

    # Read all labels
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]

    # Remove all occurrences (case-insensitive)
    new_labels = [l for l in labels if l.lower() != label.strip().lower()]

    if len(new_labels) == len(labels):
        return jsonify({'success': False, 'message': 'Label not found.'}), 404

    # Write back filtered labels
    with open(label_file, 'w', encoding='utf-8') as f:
        for l in new_labels:
            f.write(l + '\n')

    return jsonify({'success': True, 'message': 'Label removed.'})

@app_flask.route('/live_video')
def live_video():
    return render_template('live_video.html')

#@app_flask.route('/stop_camera', methods=['POST'])
#def stop_camera():
    # Release the camera
 #   global cap
  #  if cap is not None:
   #     cap.release()
    #    cap = None
    #return jsonify({'status': 'success'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host='0.0.0.0', port=port, debug=True)