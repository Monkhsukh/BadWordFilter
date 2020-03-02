import os, slackclient, time
import random
from Profanity import predict, predict_prob

# delay in seconds before checking for new events 
SOCKET_DELAY = 1
# slackbot environment variables
VALET_SLACK_NAME = os.environ.get('VALET_SLACK_NAME')
VALET_SLACK_TOKEN = os.environ.get('VALET_SLACK_TOKEN')
VALET_SLACK_ID = os.environ.get('VALET_SLACK_ID')
valet_slack_client = slackclient.SlackClient(VALET_SLACK_TOKEN)

# how the bot is mentioned on slack
def get_mention(user):
    return '<@{user}>'.format(user=user)

valet_slack_mention = get_mention(VALET_SLACK_ID)

def is_private(event):
    """Checks if private slack channel"""
    return event.get('channel').startswith('D')

def is_for_me(event):
    """Know if the message is dedicated to me"""
    # check if not my own event
    type = event.get('type')
    if type and type == 'message' and not(event.get('user')==VALET_SLACK_ID):
        # in case it is a private message return true
        #if is_private(event):
            #return True
        # in case it is not a private message check mention
        #text = event.get('text')
        #channel = event.get('channel')
        #if valet_slack_mention in text.strip().split():
            #return True
        return True

def post_message(message, channel):
    valet_slack_client.api_call('chat.postMessage', channel=channel,
                          text=message, as_user=True)

def answer(message):
    prob = predict_prob([message])[0]
    prob = round(prob,4)
    pred = 'Yes' if predict([message])[0]== 1 else 'No'

    response_template = 'filtered: {prediction}, probablity: {probablity}'
    return response_template.format(prediction=pred, probablity=prob)

def handle_message(message, user, channel):
    msg = answer(message)
    post_message(message=msg, channel=channel)

def run():
    if valet_slack_client.rtm_connect():
        print('[.] Botchat is ON...')
        while True:
            event_list = valet_slack_client.rtm_read()
            if len(event_list) > 0:
                for event in event_list:
                    print(event)
                    if is_for_me(event):
                        handle_message(message=event.get('text'), user=event.get('user'), channel=event.get('channel'))
            time.sleep(SOCKET_DELAY)
    else:
        print('[!] Connection to Slack failed.')

if __name__=='__main__':
    run()
