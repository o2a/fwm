#!/usr/bin/env python

import boto3
import time
import os
import logging
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler


# Replace these with your AWS credentials
access_key_id = ""
secret_access_key = ""
region_name = "eu-west-1"

# Replace with the directory to monitor
#target_directory = "/mnt/nerclactdb/appdev/HYDROLOGY/floating-weed-manager/mapserver/prod/mapdata/lakeO/Automatic_Classified_Maps/"

# Create SES client
ses_client = boto3.client('ses', region_name=region_name,
                         aws_access_key_id=access_key_id,
                         aws_secret_access_key=secret_access_key)

# Email configuration
SENDER = "noreply@ceh.ac.uk"
#RECIPIENT = "henry.j.thompson@kcl.ac.uk"
#recipient = ["olaawe@ceh.ac.uk", "henry.j.thompson@kcl.ac.uk"]
recipient = ["olaawe@ceh.ac.uk"]
body_text = "New Map Data available in the Floating Weed Manager Portal"
SUBJECT = "New Data - FWM Portal"
#SUBJECT = "New Data `date +\%Y\%m\%d-\%H\%M`- FWM Portal"
#BODY_TEXT = "New Classified Map available in Floating Weed Manager Portal"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class Watcher:
    #DIRECTORY_TO_WATCH = "/mnt/nerclactdb/appdev/HYDROLOGY/floating-weed-manager/mapserver/prod/mapdata/lakeO/Automatic_Classified_Maps"
    DIRECTORY_TO_WATCH = "/mnt/nerclactdb/appdev/HYDROLOGY/floating-weed-manager/mapserver/prod/mapdata"
    #DIRECTORY_TO_WATCH = "."


    def __init__(self):
        self.observer = PollingObserver()

    def run(self):
        try:
            event_handler = Handler()
            self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
            self.observer.start()
            logger.info(f"Watching directory: {self.DIRECTORY_TO_WATCH}")
            while True:
                time.sleep(5)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            self.observer.stop()
            logger.info("Observer Stopped")

        self.observer.join()

# To add the files to the body of the email, grab everything in event.src_path. Strip out all the preceeding folder names.
#  event.src_path would be passed as an argument into send_email_notification fn. It's content is to be concatenated with body_text
# If a file is created, an email is sent. How do you take all files created in a time period and put everything in the body of the email and then send???
class Handler(FileSystemEventHandler):
    @staticmethod
    def send_email_notification():
        try:
            response = ses_client.send_email(
                Destination={'ToAddresses': recipient},
                Message={
                    'Body': {'Text': {'Charset': 'UTF-8', 'Data': body_text}},
                    'Subject': {'Charset': 'UTF-8', 'Data': SUBJECT},
                },
                Source=SENDER,
            )
            logger.info(f"Email sent successfully: {response}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def on_created(self, event):
        if event.is_directory:
            return
        logger.info(f"New file created: {event.src_path}")
        self.send_email_notification()


if __name__ == '__main__':
    w = Watcher()
    w.run()