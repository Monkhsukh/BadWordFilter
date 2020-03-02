import requests
import json
from Profanity import predict, predict_prob
#from AutoReply import predictReply


response = requests.get("http://192.168.131.135:8080/ResocialFilter/v1/getComments")

if response.status_code == 200:
        print('Successfully get content')
        #print('')
        jComments = json.loads(response.text)['commentList']
        for i in jComments:
               # print(predict([i["message"]]))
                if predict([i["message"]]) == 1:
                        print(i["message"])
                        parameters = {
                                "commentId": i["comment_id"],
                                "result":"true"
                        }
                        responseDelete = requests.get("http://192.168.131.135:8080/ResocialFilter/v1/badWordsFilter", params = parameters)
                        if responseDelete.status_code == 200:
                            print("Deleted")
                        elif response.status_code == 404:
                            print('Not Found.')
 #               else:
                        # I added this code 
                        # After test this print supposed to be write in DB using API
  #                      print(i["message"])
   #                     predictReply(i["message"]) 
elif response.status_code == 404:
        print('Not Found.')
