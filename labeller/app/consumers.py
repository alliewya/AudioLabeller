from channels.generic.websocket import WebsocketConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from channels.consumer import SyncConsumer
import math

#from channels import Group


# def prog(message):
#     Group('proggroup').add(message.reply_channel)

# def prog2(message):
#     Group('proggroup').add("ayy")
#     Group('proggroup').add(message)

# class ProgressConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()

#     async def disconnect(self, close_code):
#         pass

#     async def send_progress(self, message):
#         print("sendfromconsumer")
#         # await self.send({
#         #     "type": "websocket.send",
#         #     "text": message,
#         # })
#         await self.send(text_data="eee")
#         print("sendfromconsumer2")
    
#     def sent_progress_sync(self,message):
#         print("Sync send")
#         print(message)
#         self.send(text_data="eee")
#         #self.send(text_data=json.dumps(message))

progress2 = 0

class ProgressConsumer(WebsocketConsumer):
    
    #progress = 0

    def connect(self):
        self.accept()
        self.send(text_data="json.dumps(message)")

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        # Called with either text_data or bytes_data for each frame
        # You can call:
        self.send(text_data=str(progress2))
        print("What? " + str(self.get_progress()))
        # Or, to send a binary frame:
        #self.send(bytes_data="Hello world!")
        # Want to force-close the connection? Call:
        #self.close()
        # Or add a custom WebSocket error code!
        #self.close(code=4123)
        if(progress2 == 100):
            self.send(text_data=str(progress2))
            self.close(code=1000)
        
    def get_progress(self):
         return(progress2)
    
    def update_progress(self,update):
        progress2 = math.ceil(update)
        #progress2 = self.progress
        print("Percentage 2"+ str(progress2))

    def sent_progress_sync(self,message):
        #self.channel_layer.group_send("progress", {"type": "my.message", "data": "Hello, group!"})
        # self.close()
        print("Sync send")
        # print(message)
        #self.send({"type": "websocket.send","text": "message", })

