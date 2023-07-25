import json
import threading
import schedule
from flask import Flask, render_template, Response, request, session, redirect, url_for,flash
import datetime, time
import os, sys
import face_recognition
import tensorflow as tf
import numpy as np
import cv2
import pathlib
from flask_pymongo import PyMongo
from flask import jsonify
import matplotlib.pyplot as plt
from flask_mail import Mail, Message # pip install Flask-Mail
from google.oauth2 import service_account # pip install google-auth-oauthlib
import googleapiclient.discovery # pip install google-api-python-client
import base64 # pip install base64
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


GMAIL_CREDENTIALS_FILE = 'your own gmail API json file path'
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']
GMAIL_USER_ID = 'youremail@gmail.com'
GMAIL_APP_PASSWORD = '********'


# Emotion Frequency data (use the frequency calculated earlier)
emotion_frequency = {'Neutral': 13, 'Sadness': 7, 'Happy': 3}

global capture, rec_frame, switch, face, out

# dataset folder as app.config['UPLOAD_FOLDER']
UPLOAD_FOLDER = './dataset'

known_face_encodings = []
known_face_names = []

emotion = ["Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise", "Neutral"]

model = tf.keras.models.load_model(r"./model.h5")
# get all images
# for path_name in pathlib.Path('./dataset').iterdir():
#     if path_name.is_file():  # check if it is file
#         image = face_recognition.load_image_file(path_name)  # load image
#         face_encodings = face_recognition.face_encodings(image)  # get face encodings
#         known_face_encodings.append(face_encodings[0])  # get first face encoding
#         known_face_names.append(path_name.stem)  # get file name without extension


capture = 0
switch = 1
detect = False


# make detect true every 10 seconds
def make_detect_true():
    global detect
    detect = True


schedule.every(10).seconds.do(make_detect_true)


# threading
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)


thread = threading.Thread(target=run_schedule)
thread.start()

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

# Configure Flask-Mail with Gmail SMTP settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = GMAIL_USER_ID
app.config['MAIL_PASSWORD'] = GMAIL_APP_PASSWORD

# Create the Mail object
mail = Mail(app)


# # Gmail API client
# credentials = service_account.Credentials.from_service_account_file(GMAIL_CREDENTIALS_FILE, scopes=GMAIL_SCOPES)
# gmail_service = googleapiclient.discovery.build('gmail', 'v1', credentials=credentials)


camera = cv2.VideoCapture(0)

app.secret_key = 'the quick'

MONGODB_URI = "mongodb://localhost:27017/Emotion"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Connect to MongoDB
mongo = PyMongo(app, uri=MONGODB_URI)

User = mongo.db.users  # collection name
Employee = mongo.db.employees  # collection name
EmotionRecord = mongo.db.emotionrecord  # collection name

image_capture_every = 5  # take a picture every 5 seconds

# get Employee data
employees = Employee.find()
for employee in employees:
    filename = employee['filename']
    name = employee['name']
    image = face_recognition.load_image_file("dataset/" + filename)  # load image
    face_encodings = face_recognition.face_encodings(image)  # get face encodings
    known_face_encodings.append(face_encodings[0])  # get first face encoding
    known_face_names.append(name)  # get file name without extension

print(known_face_names)


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, detect
    while True:
        success, frame = camera.read()
        if success:

            name = "Unknown"

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            # print(face_encodings)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                #  face crop
                x, y, w, h = face_locations[0]

                face = frame[face_locations[0][0] - 10:face_locations[0][2] + 20,
                       face_locations[0][3] + 20:face_locations[0][1] - 10]

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # resize frame by 48x48

                    image = cv2.resize(face, (48, 48))

                    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # convert image to grayscale
                    x = np.expand_dims(img_gray, axis=-1)  # add channel dimension
                    x = x / 255.0  # rescale the pixel values to [0, 1]
                    x = np.expand_dims(x, axis=0)  # add batch dimension
                    preds = model.predict(x)
                    class_num = np.argmax(preds, axis=1)[0]

                    frame = cv2.putText(frame, emotion[class_num], (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 4)
                    if detect:
                        print('record saved' + str(datetime.datetime.now()) + ',' + name + ',' + emotion[class_num])
                        # 0, 65 means x,y coordinates
                        frame = cv2.putText(frame, 'record saved', (200, 65), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 4)
                        EmotionRecord.insert_one(
                            {'name': name, 'emotion': emotion[class_num], 'time': str(datetime.datetime.now())}
                        )

                        detect = False

            frame = cv2.putText(frame, name, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        if success:
            if (capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/download_data')
def download_emotion_records_in_json():
    records = EmotionRecord.find()
    json_data = []
    for record in records:
        json_data.append({'name': record['name'], 'emotion': record['emotion'], 'time': record['time']})
    json.dump(json_data, open('records.json', 'w'))
    print(json_data)
    return render_template('download_data.html', error='',emotion_data=json_data)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    # check if user is already logged in
    login_ = False
    if 'username' in session:
        login_ = True
    return render_template('home.html',login = login_)


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        return render_template('dashboard.html')
    return render_template('dashboard.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # check if user is already logged in
    if 'username' in session:
        return redirect('/dashboard')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username and password match in the database
        user = User.find_one({'username': username, 'password': password})
        # user = {'username': username, 'password': password}
        if user:
            session['username'] = username  # Store the username in the session
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html', error='')


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']

        return render_template('dashboard.html', username=username)
        # return render_template('index.html', username=username)
    else:
        return redirect('/login')


@app.route('/employee_data_upload', methods=['GET', 'POST'])
def employee_data_upload():
    # check enctype
    print(request.content_type)
    if request.method == 'POST' and request.content_type.startswith('multipart/form-data'):
        file_ = request.files['file']
        time_ = datetime.datetime.now()
        time_ = str(time_)
        filename = "pic" + time_ + file_.filename

        file_.save(os.path.join("dataset", filename))
        return render_template('/employee_data_upload.html', success='File uploaded successfully', filename=filename)
    elif request.method == "POST":
        # filename check
        form_data = request.form
        name = form_data['name']
        email = form_data['email']
        phone = form_data['phone']
        designation = form_data['designation']
        company = form_data['company']
        filename = form_data['filename']
        Employee.insert_one(
            {'name': name, 'email': email, 'phone': phone, 'designation': designation, 'company': company,
             'filename': filename})
        return render_template('/employee_data_upload.html', success='Data uploaded successfully')

    return render_template('/employee_data_upload.html', success='')


@app.route('/logout')
def logout():
    # create static url for static files
    session.pop('username', None)
    return redirect('/')


@app.route('/analytics')
def analytics():
    # remove images folder if exists
    if os.path.exists('static/images'):
        os.system('rm -rf static/images')

    # create images folder
    os.mkdir('static/images')

    # get emotion frequency data from database for each employee
    records = EmotionRecord.find()

    # employee name -> emotion list
    employees_dict = dict()
    for record in records:
        if record['name'] in employees_dict:
            employees_dict[record['name']].append(record['emotion'])
        else:
            employees_dict[record['name']] = [record['emotion']]

    # assign 0 to each emotion for each employee
    # employee name -> emotion -> frequency
    employees_emotion_frequency = dict()
    for employee in employees_dict.keys():
        employees_emotion_frequency[employee] = dict()
        for e in emotion:
            employees_emotion_frequency[employee][e] = 0

    # calculate frequency for each employee
    # employee name -> emotion -> frequency
    for employee in employees_dict.keys():
        for e in employees_dict[employee]:
            employees_emotion_frequency[employee][e] += 1

    graphs = dict()
    # create folder for each employee name
    for employee in employees_emotion_frequency.keys():
        os.mkdir('static/images/' + employee)
        graphs[employee] = dict()

        employee_emotions = employees_emotion_frequency[employee]
        # Bar graph
        plt.figure(figsize=(8, 5))
        plt.bar(employee_emotions.keys(), employee_emotions.values(),
                color=['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink'])
        plt.xlabel('Emotion')
        plt.ylabel('Frequency')
        plt.title('Emotion Frequency Bar Graph')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # put image in html
        plt.savefig('static/images/' + employee + '/bar.png')

        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(employee_emotions.values(), labels=employee_emotions.keys(), autopct='%1.1f%%',
                colors=['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink'])
        plt.title('Emotion Frequency Pie Chart')
        plt.axis('equal')
        plt.tight_layout()

        # put image in html
        plt.savefig('static/images/' + employee + '/pie.png')
        pie_graph = url_for('static', filename='images/' + employee + '/pie.png')
        bar_graph = url_for('static', filename='images/' + employee + '/bar.png')
        graphs[employee]['pie'] = pie_graph
        graphs[employee]['bar'] = bar_graph

    return render_template('analytics.html', graphs=graphs)


# contactus
@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

def get_gmail_service():
    creds = None
    if os.path.exists('./client_secret_845112444690-47g01ekcsegu9brjfnd6b7v6i6fol378.apps.googleusercontent.com.json'):
        creds = Credentials.from_authorized_user_file('./client_secret_845112444690-47g01ekcsegu9brjfnd6b7v6i6fol378.apps.googleusercontent.com.json', GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                './client_secret_845112444690-47g01ekcsegu9brjfnd6b7v6i6fol378.apps.googleusercontent.com.json', GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open('./client_secret_845112444690-47g01ekcsegu9brjfnd6b7v6i6fol378.apps.googleusercontent.com.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

@app.route('/send_email', methods=['POST'])
def send_email():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    print(name, email, message)

    if not name or not email or not message:
        flash('Please fill out all the fields.', 'error')
        return redirect("/contactus")

    try:
        service = get_gmail_service()

        # Compose the email
        subject = f"Contact Us Form - {name}"
        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        raw_message = f"From: {email}\nTo: {GMAIL_USER_ID}\nSubject: {subject}\n\n{body}"
        raw_message_bytes = base64.urlsafe_b64encode(raw_message.encode('utf-8'))
        raw_message_str = raw_message_bytes.decode('utf-8')

        # Send the email
        service.users().messages().send(userId='me', body={'raw': raw_message_str}).execute()

        flash('Your message has been sent successfully.', 'success')
        return redirect("/contactus")

    except Exception as e:
        print("some thing went wrong")
        flash('An error occurred while sending the email. Please try again later.', 'error')
        return redirect("/contactus")

@app.route('/support')
def support():
    return render_template('support.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form_data = request.form
    if request.method == 'POST':
        username = form_data['username']
        password = form_data['password']
        email = form_data['email']
        company = form_data['company']
        phone = form_data['phone']
        User.insert_one(
            {'username': username, 'password': password, 'email': email, 'company': company, 'phone': phone})
        return redirect('/')
    return render_template('signup.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/inventors')
def inventors():
    return render_template('inventors.html')


@app.route('/terms_of_services')
def terms_of_services():
    return render_template('terms_of_services.html')


@app.route('/works')
def works():
    return render_template('works.html')


@app.route('/view')
def view():
    employees  = Employee.find()

    return render_template('view.html',employees=employees)


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
