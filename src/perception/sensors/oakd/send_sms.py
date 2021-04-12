# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'AC9298d5d2c6bec5608bf477b58cf57d83'
auth_token = '45017941c09868091e2800eaa4f000b4'
client = Client(account_sid, auth_token)

message = client.messages \
    .create(
         body='Hi Kuttiboss, you are great. (sending from our vision project)',
         from_='+17148315421',
         to='+17064103402'
     )

print(message.sid)