from flask_mail import Mail, Message # pip install Flask-Mail
from google.oauth2 import service_account # pip install google-auth-oauthlib
import googleapiclient.discovery # pip install google-api-python-client
import base64 # pip install base64

GMAIL_CREDENTIALS_FILE = 'insert your own gmail API json file'
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']
GMAIL_USER_ID = 'youremail@gmail.com'
GMAIL_APP_PASSWORD = '*********'

app = Flask(__name__)

# Configure Flask-Mail with Gmail SMTP settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = GMAIL_USER_ID
app.config['MAIL_PASSWORD'] = GMAIL_APP_PASSWORD

# Create the Mail object
mail = Mail(app)


# Gmail API client
credentials = service_account.Credentials.from_service_account_file(GMAIL_CREDENTIALS_FILE, scopes=GMAIL_SCOPES)
gmail_service = googleapiclient.discovery.build('gmail', 'v1', credentials=credentials)


# Home route to display the contact form
@app.route('/', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Basic form validation
        if not name or not email or not message:
            flash('All fields are required.', 'error')
            return redirect(url_for('contact'))

        try:
            # Create the email message
            msg = Message('Contact Form Submission', recipients=['afzalkam011@gmail.com'])  # Replace with your desired email address
            msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"

            # Send the email
            mail.send(msg)
            flash('Thank you! Your message has been sent.', 'success')
        except Exception as e:
            print(str(e))
            flash('Something went wrong. Please try again later.', 'error')

        return redirect(url_for('contact'))

    return render_template('contact_form.html')

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Replace with your secret key for flashing messages
    app.run(debug=True)
