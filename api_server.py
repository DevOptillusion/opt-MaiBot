from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import time
import json
import os
import sys
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env", override=True)

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="MaiM Bot API Server")

class ChatRequest(BaseModel):
    message: str
    user_id: str
    username: str
    nickname: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None

class ChatResponse(BaseModel):
    response: str
    success: bool = True

# Global variables to store bot instances
chat_bot = None
main_system = None
captured_replies = {}  # Store captured replies by message_id

# Store mapping: bot's user_id (username) -> Discord's numeric user_id
discord_user_id_mapping = {}
WEB_API_URL = os.getenv("WEB_API_URL", "http://web-api:8000")

@app.on_event("startup")
async def startup_event():
    """Initialize the MaiM Bot system on startup"""
    global chat_bot, main_system
    
    try:
        print("Initializing MaiM Bot API Server...")
        
        # Import the bot modules
        from src.chat.message_receive.bot import ChatBot
        from src.main import MainSystem
        from maim_message import Seg, UserInfo, BaseMessageInfo
        from src.chat.message_receive.message import MessageRecv
        from src.chat.message_receive.chat_stream import get_chat_manager
        
        # Initialize the main system
        main_system = MainSystem()
        await main_system._init_components()
        
        # Get the chat bot instance
        chat_bot = ChatBot()
        await chat_bot._ensure_started()
        
        # Patch message sender to intercept Discord messages
        _patch_discord_message_sender()
        
        print("MaiM Bot API Server initialized successfully!")
        
    except Exception as e:
        print(f"Failed to initialize MaiM Bot API Server: {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    return {"message": "MaiM Bot API Server is running", "status": "ready"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if chat_bot and main_system:
        return {"status": "healthy", "bot_initialized": True}
    else:
        return {"status": "unhealthy", "bot_initialized": False}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the MaiM Bot system"""
    print(f"[CORE-API] /chat endpoint called!", flush=True)
    global chat_bot, main_system
    
    if not chat_bot or not main_system:
        raise HTTPException(status_code=503, detail="Bot system not initialized")
    
    try:
        print(f"API Server received message from {request.nickname or request.username} (username: {request.username}): {request.message}")
        
        # Store user_id mapping: bot uses username, Discord bridge needs numeric ID
        discord_user_id_mapping[request.username] = request.user_id
        
        # Import required modules
        from maim_message import Seg, UserInfo, BaseMessageInfo, GroupInfo
        from src.chat.message_receive.message import MessageRecv
        from src.chat.message_receive.chat_stream import get_chat_manager
        from src.plugin_system.apis import generator_api
        
        # Create user info
        # For Discord, use username as the primary identifier instead of user_id
        user_info = UserInfo(
            user_id=request.username,  # Use username as user_id for Discord
            user_nickname=request.nickname or request.username,  # Use nickname if available, fallback to username
            platform="discord"  # Use discord as platform
        )
        
        # Create message segment
        message_segment = Seg(type="text", data=request.message)
        
        # Create format info with text support
        from maim_message import FormatInfo
        format_info = FormatInfo(
            content_format=["text"],
            accept_format=["text", "emoji", "reply"]
        )
        
        # Set group_info for specific Discord channel
        group_info = None
        if str(request.channel_id) == "1393102620624687187":
            group_info = GroupInfo(
                platform="discord",
                group_id=request.channel_id,
                group_name=request.channel_name or "chat-with-ada",
            )

        # Generate unique message ID in the same format as the bot (send_api format)
        timestamp = time.time()
        message_id = f"send_api_{int(timestamp * 1000)}"
        
        print(f"API Server generating message_id: {message_id}")
        
        message_info = BaseMessageInfo(
            platform="discord",
            message_id=message_id,
            time=timestamp,
            group_info=group_info,  # Now None
            user_info=user_info,
            format_info=format_info
        )
        
        # Create the message dictionary
        message_dict = {
            "message_info": message_info.to_dict(),
            "message_segment": message_segment.to_dict(),
            "raw_message": request.message,
            "processed_plain_text": request.message,
            "detailed_plain_text": request.message
        }
        
        # Create MessageRecv object
        message = MessageRecv(message_dict)
        
        # Get or create chat stream (no group_info)
        chat = await get_chat_manager().get_or_create_stream(
            platform="discord",
            user_info=user_info,
            group_info=group_info
        )
        
        message.update_chat_stream(chat)
        
        # Process the message through the bot
        await chat_bot.message_process(message_dict)
        
        # Process the message through the bot (replies will be sent via HTTP to web-api)
        # No need to wait for replies here since they're handled by the HTTP system
        # Return empty response since actual replies are sent via web-api
        response_text = ""
                
        return ChatResponse(response=response_text, success=True)
        
    except Exception as e:
        print(f"Error processing message in API Server: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response=f"Sorry, I encountered an error processing your message: {str(e)}", 
            success=False
        )

@app.post("/message", response_model=ChatResponse)
async def process_message(request: Dict[str, Any]):
    """Process a message in the format expected by the bot system"""
    global chat_bot, main_system
    
    if not chat_bot or not main_system:
        raise HTTPException(status_code=503, detail="Bot system not initialized")
    
    try:
        print(f"API Server received message data: {json.dumps(request, indent=2)}")
        
        # Process the message through the bot
        await chat_bot.message_process(request)
        
        return ChatResponse(
            response="Message processed successfully by MaiM Bot system",
            success=True
        )
        
    except Exception as e:
        print(f"Error processing message data in API Server: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response=f"Error processing message: {str(e)}",
            success=False
        )

def _patch_discord_message_sender():
    """Patch _send_message to intercept Discord messages and send to web-api via HTTP"""
    try:
        import src.chat.message_receive.uni_message_sender as uni_module
        original_send = uni_module._send_message
        
        from src.chat.message_receive.message import MessageSending
        from maim_message import Seg
        
        async def patched_send(message: MessageSending, show_log=True) -> bool:
            if not (message.message_info and message.message_info.platform == "discord"):
                return await original_send(message, show_log)
            
            try:
                bot_user_id = message.chat_stream.user_info.user_id if message.chat_stream and message.chat_stream.user_info else None
                discord_user_id = discord_user_id_mapping.get(bot_user_id) if bot_user_id else bot_user_id
                
                content = ""
                message_type = "text"
                
                if isinstance(message.message_segment, Seg):
                    seg = message.message_segment
                    if seg.type == "text":
                        content = str(seg.data) if seg.data else ""
                    elif seg.type == "emoji":
                        message_type = "emoji"
                        content = str(seg.data) if seg.data else ""
                    elif seg.type == "seglist":
                        content = "".join([str(sub_seg.data) for sub_seg in seg.data if isinstance(sub_seg, Seg) and sub_seg.type == "text"])
                    else:
                        content = str(seg.data) if seg.data else ""
                
                if not content and hasattr(message, 'processed_plain_text'):
                    content = message.processed_plain_text
                
                if not content:
                    return True
                
                payload = {
                    "platform": "discord",
                    "user_id": discord_user_id,
                    "content": content,
                    "message_type": message_type
                }
                
                if message.message_info.group_info:
                    payload["channel_id"] = str(message.message_info.group_info.group_id)
                
                print(f"[Discord] Sending to web-api: user_id={discord_user_id}, content={content[:50]}...")
                response = requests.post(f"{WEB_API_URL}/send_message", json=payload, timeout=10)
                
                if response.status_code == 200:
                    print(f"[Discord] Successfully sent to web-api")
                    return True
                else:
                    return await original_send(message, show_log)
            except Exception as e:
                print(f"[Discord] Error: {e}")
                return await original_send(message, show_log)
        
        uni_module._send_message = patched_send
        print("✓ Patched Discord message sender")
    except Exception as e:
        print(f"✗ Failed to patch: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 
