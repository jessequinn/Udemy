import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv, find_dotenv


def send_email(email, height, avg_height, count):
    load_dotenv(find_dotenv())
    from_email = os.getenv("EMAIL")
    from_password = os.getenv("PASSWORD")
    to_email = email

    subject = "Height data"
    message = "Hey there, your height is <strong>%s</strong>. Average height of <strong>%s</strong> is <strong>%s</strong>." % (
    height, count, avg_height)

    msg = MIMEText(message, "html")
    msg['Subject'] = subject
    msg['To'] = to_email
    msg['From'] = from_email

    gmail = smtplib.SMTP('smtp.gmail.com', 587)
    gmail.ehlo()
    gmail.starttls()
    gmail.login(from_email, from_password)
    gmail.send_message(msg)


if __name__ == '__main__':
    send_email('me@jessequinn.info', 56)
