"""
    Simple ChatBot for Slack

    Author: Ada Kaminkure
    A.K.A: ada_92
"""

import time
import os
from random import randrange
from slackclient import SlackClient

# Constant Variables
SLACK_CLIENT = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

print(os.environ.get('SLACK_BOT_TOKEN'))

BOT_ID = SLACK_CLIENT.api_call("auth.test")["user_id"]
CHANNEL_LIST = SLACK_CLIENT.api_call("channels.list").get("channels", [])
ANSWERS_LIST = [
    "Umm, interesting I will ask my master for its meaning",
    "I think it cool",
    "Howdy!",
    "What's up?",
    "I see",
    "Anythings else?",
    "I'm done!!!"
]


def find_channels_id(channel_name, channel_list):
    """
        Find channel id by its name
        channel_name (string): name of finding channel
        channel_list (list): all channel list in the workspace
        return -> channel_id
    """
    match = [ch.get("id") for ch in channel_list if ch.get("name") == channel_name]
    return match[0]



if __name__ == "__main__":
    
    # Running Bot in this section

    if SLACK_CLIENT.rtm_connect(with_team_state=False):
        print("I'm online, please order me...")
        response_channels = find_channels_id("some-channel", CHANNEL_LIST)
        SLACK_CLIENT.api_call(
            "chat.postMessage",
            channel=response_channels,
            text="Hello, My name is ICSCO-BOT",
        )
        while True:
            data = SLACK_CLIENT.rtm_read()
            if not data:
                continue
            else:
                new_data = data[0]
                print(new_data)
                if new_data.get("type") == "message" \
                    and new_data.get("subtype") != "bot_message":

                    SLACK_CLIENT.api_call(
                        "chat.postMessage",
                        channel=response_channels,
                        text=ANSWERS_LIST[randrange(len(ANSWERS_LIST))]
                    )

            time.sleep(1)
    else:
        print("Something wrong, please check your internet connection!")
