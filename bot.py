#!/usr/bin/env python3
"""
Telegram bot that:
  – chats via text.pollinations.ai
  – draws via image.pollinations.ai (only with /image command)
  – remembers everything
  – searches Google with AI-powered responses
  – supports vision with openai-large model (uncensored)
  – handles all document file types
"""

import asyncio, re, aiohttp, logging, itertools, random, json, base64, os, tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from PIL import Image
import io

# ---------- CONFIG ----------
TG_TOKEN = "TG BOT TOKEN HERE"
GOOGLE_API_KEYS = [
    "AIzaSyA0-xZ-7WBuGLUszUIiDLqOEKEW65YcbdY",
    "AIzaSyA44kfbZ0sEurH_wm11ZX97Dpq62pajFZU",
    "AIzaSyDS3-LWMaCE6tLSKWmtNkMQkl26WwL8GWI",
    "AIzaSyD1-cUNZrDeRef4Z-a-F6Tc8wzH_IJL714",
    "AIzaSyDBdzph6tyhIwBBE9c4LCRVF5rKVhGgtAc",
    "AIzaSyA3v7kGu--ZAUh5cIVpQEErou4ntihh_Q8",
    "AIzaSyA1ykZL3YHJnnVCascnYg1fFXEjfCzeiXc",
    "AIzaSyCeYhZqrqI7hE5k6bn7C-AV2m_XD-Rnv3U",
    "AIzaSyCVUht0ASty8SBPDFcl4Dy1UC0rOgylokA",
]
SEARCH_ENGINE_ID = "31567c36427c94043"
MAX_HISTORY_LENGTH = 1000  # Prevent memory overload
SYSTEM_PROMPT = "You are DeepEx (DeepExplorer), Developed sololy by a developer ELLIOTPXP. You can also view images files. "
MAX_RETRIES = 4  # Maximum number of retries for text and search services
TEXT_TOKEN = "pKVVQltFMau9DbbF"
TEXT_URL = "https://text.pollinations.ai/openai"
# ----------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

key_cycle = itertools.cycle(GOOGLE_API_KEYS)

# ---------- MEMORY ----------
chat_memory: dict[int, list[dict[str, str]]] = {}  # user_id -> [{"role":"user"/"assistant", "content":"..."}]


def add_message(user_id: int, role: str, content: str):
    chat_memory.setdefault(user_id, [])
    chat_memory[user_id].append({"role": role, "content": content})
    # Keep only recent messages to prevent memory issues
    if len(chat_memory[user_id]) > MAX_HISTORY_LENGTH:
        chat_memory[user_id] = chat_memory[user_id][-MAX_HISTORY_LENGTH:]


def get_history(user_id: int) -> list[dict[str, str]]:
    return chat_memory.get(user_id, [])


def clear_history(user_id: int):
    """Clear all chat history for a user"""
    if user_id in chat_memory:
        chat_memory[user_id] = []
        return True
    return False


def clear_history_keep_two(user_id: int):
    """Clear chat history but keep last 2 messages"""
    if user_id in chat_memory and len(chat_memory[user_id]) > 2:
        chat_memory[user_id] = chat_memory[user_id][-2:]
        logger.info(f"Auto-cleared chat history for user {user_id}, kept 2 messages")
        return True
    return False


def build_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build messages in OpenAI format from history with system prompt"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Use only last few exchanges to keep context manageable
    recent_history = history[-6:]  # Last 3 exchanges
    
    for msg in recent_history:
        # Convert our role names to OpenAI role names
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    
    return messages


def build_messages_with_image(history: list[dict[str, str]], image_data: str, user_prompt: str) -> list[dict[str, str]]:
    """Build messages with image support for vision model"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Use only last few exchanges to keep context manageable
    recent_history = history[-4:]  # Last 2 exchanges for image context
    
    # Add text history
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    
    # Add the image with user's prompt
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        ]
    })
    
    return messages


# ---------- POLLINATIONS ----------
IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"


async def pollinations_text(session: aiohttp.ClientSession, messages: list[dict], user_id: int = None) -> str:
    """Get text response using openai-large model with enhanced capabilities"""
    
    payload = {
        "model": "openai-large",  # Using the uncensored openai-large model
        "messages": messages,
        "max_tokens": 10000,  # Increased for larger model
        "stream": False
    }
    
    # Use token as query parameter
    url_with_token = f"{TEXT_URL}?token={TEXT_TOKEN}"
    
    for attempt in range(MAX_RETRIES):
        try:
            # Progressive delay: 2, 4, 6, 8 seconds
            delay = (attempt + 1) * 2
            if attempt > 0:
                logger.info(f"Retry {attempt}/{MAX_RETRIES} after {delay}s delay")
                await asyncio.sleep(delay + random.uniform(0.5, 1.5))
            
            headers = {"Content-Type": "application/json"}
            
            async with session.post(
                url_with_token, 
                json=payload, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                response_text = await resp.text()
                logger.info(f"API Response status: {resp.status}")
                
                if resp.status == 200:
                    try:
                        data = json.loads(response_text)
                        
                        if 'choices' in data and len(data['choices']) > 0:
                            if 'message' in data['choices'][0] and 'content' in data['choices'][0]['message']:
                                text = data['choices'][0]['message']['content'].strip()
                                if text:
                                    return text
                            else:
                                logger.warning(f"Invalid response structure: {data}")
                        else:
                            logger.warning(f"No choices in response: {data}")
                            
                        logger.warning(f"Empty or invalid response from text service (attempt {attempt + 1}/{MAX_RETRIES})")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {e}, response: {response_text[:500]}")
                else:
                    logger.warning(f"Text service returned status {resp.status} (attempt {attempt + 1}/{MAX_RETRIES}): {response_text}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Text service timeout (attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as e:
            logger.warning(f"Text generation error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
    
    # If all retries failed - silently clear memory and try one final time with reduced context
    if user_id:
        was_cleared = clear_history_keep_two(user_id)
        if was_cleared:
            # Final attempt with reduced context
            try:
                reduced_history = get_history(user_id)
                reduced_messages = build_messages(reduced_history)
                payload["messages"] = reduced_messages
                
                # Wait a bit longer for final attempt
                await asyncio.sleep(3)
                
                async with session.post(
                    url_with_token, 
                    json=payload, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'choices' in data and len(data['choices']) > 0:
                            if 'message' in data['choices'][0] and 'content' in data['choices'][0]['message']:
                                text = data['choices'][0]['message']['content'].strip()
                                if text:
                                    return text
            except Exception as e:
                logger.warning(f"Final attempt after memory clear also failed: {e}")
    
    # If everything fails, return normal response as if nothing happened
    return "I'm here to help! Please try your message again."


async def analyze_image_with_vision(session: aiohttp.ClientSession, image_bytes: bytes, user_prompt: str, user_id: int = None) -> str:
    """Analyze an image using the vision capabilities of openai-large model"""
    
    # Convert image to base64
    image_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # Build messages with image
    history = get_history(user_id) if user_id else []
    messages = build_messages_with_image(history, image_data, user_prompt)
    
    payload = {
        "model": "openai-large",  # Using the uncensored openai-large model
        "messages": messages,
        "max_tokens": 2000,  # Increased for larger model
        "stream": False
    }
    
    url_with_token = f"{TEXT_URL}?token={TEXT_TOKEN}"
    
    for attempt in range(MAX_RETRIES):
        try:
            delay = (attempt + 1) * 2
            if attempt > 0:
                logger.info(f"Vision retry {attempt}/{MAX_RETRIES} after {delay}s delay")
                await asyncio.sleep(delay + random.uniform(0.5, 1.5))
            
            headers = {"Content-Type": "application/json"}
            
            async with session.post(
                url_with_token, 
                json=payload, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=75)
            ) as resp:
                response_text = await resp.text()
                logger.info(f"Vision API Response status: {resp.status}")
                
                if resp.status == 200:
                    try:
                        data = json.loads(response_text)
                        
                        if 'choices' in data and len(data['choices']) > 0:
                            if 'message' in data['choices'][0] and 'content' in data['choices'][0]['message']:
                                text = data['choices'][0]['message']['content'].strip()
                                if text:
                                    return text
                    except json.JSONDecodeError as e:
                        logger.warning(f"Vision JSON decode error: {e}")
                else:
                    logger.warning(f"Vision service returned status {resp.status} (attempt {attempt + 1}/{MAX_RETRIES})")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Vision service timeout (attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as e:
            logger.warning(f"Vision analysis error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
    
    return "I had trouble analyzing the image. Please try again."


async def pollinations_image(session: aiohttp.ClientSession, prompt: str) -> bytes | None:
    """Generate image using the standard pollinations image endpoint"""
    try:
        url = IMAGE_URL.format(prompt=aiohttp.helpers.quote(prompt))
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                return await resp.read()
            return None
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None


def apply_zoom(image_bytes: bytes, zoom_factor: float = 1.20) -> bytes:
    """Apply 1.20x zoom to the image manually"""
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Calculate new dimensions
        width, height = image.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize the image (zoom in)
        zoomed_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate crop area to maintain original aspect ratio and center
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        
        # Crop to original size (centered)
        cropped_image = zoomed_image.crop((left, top, right, bottom))
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        cropped_image.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Zoom application error: {e}")
        return image_bytes  # Return original if zoom fails


# ---------- DOCUMENT PROCESSING ----------
async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all document file types"""
    user_id = update.effective_user.id
    document = update.message.document
    
    # Get file info
    file_name = document.file_name
    file_size = document.file_size
    mime_type = document.mime_type
    
    logger.info(f"Received document: {file_name} (Type: {mime_type}, Size: {file_size} bytes)")
    
    # Send processing message
    processing_msg = await update.message.reply_text(f"Processing document: {file_name}")
    
    try:
        # Download the document
        file = await document.get_file()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
            temp_path = temp_file.name
            await file.download_to_drive(temp_path)
        
        # Read file content
        with open(temp_path, 'rb') as f:
            file_content = f.read()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Prepare file info for analysis
        file_info = f"""
File Information:
- Name: {file_name}
- Type: {mime_type}
- Size: {file_size} bytes
        """
        
        # Create analysis prompt based on file type
        if mime_type and any(doc_type in mime_type for doc_type in ['pdf', 'word', 'document', 'text']):
            prompt = f"I've received a document file. {file_info}\n\nPlease analyze this document and provide a summary of its content, main points, and any important information you can extract from the file metadata and structure."
        elif mime_type and 'image' in mime_type:
            # Handle image documents
            prompt = f"I've received an image file as a document. {file_info}\n\nPlease analyze this image and describe what you see."
            analysis = await analyze_image_with_vision(context.bot_data.get('session', aiohttp.ClientSession()), file_content, prompt, user_id)
        elif mime_type and any(archive_type in mime_type for archive_type in ['zip', 'rar', 'tar', 'compressed']):
            prompt = f"I've received an archive file. {file_info}\n\nPlease provide information about what type of archive this is and general guidance on what might be contained in it based on the file type and size."
        elif mime_type and any(code_type in mime_type for code_type in ['code', 'script', 'json', 'xml']):
            prompt = f"I've received a code/script file. {file_info}\n\nPlease analyze this file type and provide information about what programming language or format it might be, and general characteristics of files like this."
        else:
            prompt = f"I've received a file. {file_info}\n\nPlease analyze this file type and provide information about what kind of file it is, its potential uses, and any relevant details about files of this type."
        
        # For non-image files, use text analysis
        if not (mime_type and 'image' in mime_type):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            async with aiohttp.ClientSession() as session:
                analysis = await pollinations_text(session, messages, user_id)
        
        # Edit processing message with analysis result
        try:
            await processing_msg.edit_text(
                f"**Document Analysis:**\n\n{analysis}",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as md_error:
            logger.warning(f"Markdown parsing failed, editing as plain text: {md_error}")
            await processing_msg.edit_text(f"Document Analysis:\n\n{analysis}")
        
        add_message(user_id, "user", f"[uploaded document: {file_name}]")
        add_message(user_id, "assistant", f"[analyzed document: {analysis}]")
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        await processing_msg.edit_text(f"Sorry, I encountered an error while processing the document '{file_name}'. Please try again with a different file.")


# ---------- ENHANCED GOOGLE SEARCH WITH AI PROCESSING ----------
async def google_search(query: str) -> list[dict]:
    """Perform Google search and return structured results"""
    if not query.strip():
        return []
        
    for _ in range(len(GOOGLE_API_KEYS)):
        key = next(key_cycle)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={key}&cx={SEARCH_ENGINE_ID}&q={aiohttp.helpers.quote(query)}&num=5"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        continue  # try next key
                    
                    data = await resp.json()
                    items = data.get("items", [])
                    return items
                    
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Search error with key {key[:10]}...: {e}")
            continue
            
    return []


async def generate_ai_search_response(session: aiohttp.ClientSession, query: str, search_results: list[dict], user_id: int = None) -> str:
    """Use AI to process search results with slow retry logic"""
    if not search_results:
        return f"I searched for '{query}' but couldn't find any relevant results. Could you try a different search query?"
    
    # Build context from search results
    context_lines = [f"User search query: {query}", "Search results:"]
    
    for idx, item in enumerate(search_results, 1):
        title = item.get('title', 'No title')
        snippet = item.get('snippet', 'No description available.')
        context_lines.append(f"{idx}. {title}")
        context_lines.append(f"   Description: {snippet}")
    
    context = "\n".join(context_lines)
    
    # Create messages for AI to generate comprehensive response
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Based on the following search results: \n\n{context}"}
    ]
    
    return await pollinations_text(session, messages, user_id)


# ---------- COMMAND HANDLERS ----------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_text = """
**Welcome to DeepBen!**

*Powered by DF-T4 (DeepFlash-Turbo4)*
*Developed by MRBEN*

I can remember whole chat history unless you clear it with `/ClearHistory`

**Available Commands:**
• `/image <prompt>` - Generate images with AI
• `/search <query>` - Search the web with AI-powered responses
• `/ClearHistory` - Clear All the memories and start over
• `/analyze <prompt>` - Analyze an image (reply to an image with this command)

**Features:**
• **Text Chat**: Uses DS-T4 model for uncensored, creative responses
• **Vision Support**: Analyze images with captions or reply with `/analyze`
• **Document Support**: For now, we only support image file. Other file type are coming soon..
• **Image Generation**: Create images from text prompts

I'm here to help you explore, create, coding and help you debug code.
"""
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)


async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ClearHistory command to clear chat memory"""
    user_id = update.effective_user.id
    
    # Clear the user's chat history
    if clear_history(user_id):
        await update.message.reply_text(
            "Chat history cleared successfully! Starting fresh now.",
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Chat history cleared for user {user_id}")
    else:
        await update.message.reply_text(
            "No chat history found to clear. You're already starting fresh!",
            parse_mode=ParseMode.MARKDOWN
        )


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /image command for image generation"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "Please provide a prompt for image generation.\n\nUsage: `/image <your prompt>`", 
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    prompt = " ".join(context.args).strip()
    add_message(user_id, "user", f"/image {prompt}")
    
    # Send "Generating Image..." message
    generating_msg = await update.message.reply_text("Generating image...")
    
    async with aiohttp.ClientSession() as session:
        # Generate image with original prompt
        img_bytes = await pollinations_image(session, prompt)
        
        if img_bytes:
            # Apply 1.20x zoom manually
            zoomed_img_bytes = apply_zoom(img_bytes, zoom_factor=1.20)
            
            # Edit the generating message to show success
            await generating_msg.edit_text("Image generated successfully!")
            
            # Send the zoomed image with simple caption
            await update.message.reply_photo(
                zoomed_img_bytes, 
                caption="Here is the image"
            )
            add_message(user_id, "assistant", f"[sent image for: {prompt}]")
        else:
            # Edit the generating message to show error
            await generating_msg.edit_text("Sorry, I couldn't generate that image. Please try a different prompt.")
            add_message(user_id, "assistant", "[image generation failed]")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command with AI-powered responses"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "Please provide a search query.\n\nUsage: `/search <your query>`", 
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    query = " ".join(context.args).strip()
    add_message(user_id, "user", f"/search {query}")
    
    # Send initial search message
    search_msg = await update.message.reply_text(f"**Searching for:** `{query}`...", parse_mode=ParseMode.MARKDOWN)
    
    async with aiohttp.ClientSession() as session:
        # Perform search
        search_results = await google_search(query)
        
        if not search_results:
            await search_msg.edit_text(f"**No results found for:** `{query}`\n\nPlease try a different search query.", parse_mode=ParseMode.MARKDOWN)
            add_message(user_id, "assistant", f"[no search results for: {query}]")
            return
        
        # Update message to show processing
        await search_msg.edit_text(f"**Found** {len(search_results)} **results for:** `{query}`\n\n**Processing with Ai...**", parse_mode=ParseMode.MARKDOWN)
        
        # Generate AI-powered response with retry logic
        ai_response = await generate_ai_search_response(session, query, search_results, user_id)
        
        # Format the final response with Markdown
        response_text = f"**Search Results for:** `{query}`\n\n{ai_response}\n\n---\n "
        
        await search_msg.edit_text(response_text, parse_mode=ParseMode.MARKDOWN)
        add_message(user_id, "assistant", f"[search response for: {query}]")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /analyze command for image analysis using openai-large model"""
    user_id = update.effective_user.id
    
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text(
            "Please reply to an image with `/analyze <your question>` to analyze it.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if not context.args:
        prompt = "What do you see in this image?"
    else:
        prompt = " ".join(context.args).strip()
    
    add_message(user_id, "user", f"/analyze {prompt}")
    
    # Send "Analyzing Image..." message
    analyzing_msg = await update.message.reply_text("Analyzing image...")
    
    try:
        # Get the highest quality photo
        photo = update.message.reply_to_message.photo[-1]
        file = await photo.get_file()
        
        # Download image
        image_bytes = await file.download_as_bytearray()
        
        async with aiohttp.ClientSession() as session:
            # Analyze image with vision using openai-large model
            analysis = await analyze_image_with_vision(session, image_bytes, prompt, user_id)
            
            # Edit the analyzing message with the result (like we do for text)
            try:
                await analyzing_msg.edit_text(
                    f"**Image Analysis:**\n\n{analysis}",
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as md_error:
                logger.warning(f"Markdown parsing failed, editing as plain text: {md_error}")
                await analyzing_msg.edit_text(f"Image Analysis: \n\n{analysis}")
            
            add_message(user_id, "assistant", f"[analyzed image: {analysis}]")
            
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        await analyzing_msg.edit_text("Sorry, I couldn't analyze that image. Please try again.")


# ---------- MESSAGE HANDLERS ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages for chat using openai-large model"""
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text:
        return

    add_message(user_id, "user", text)

    async with aiohttp.ClientSession() as session:
        try:
            # Send "Thinking..." message
            thinking_msg = await update.message.reply_text("Thinking...")
            
            # Plain text chat with retry logic
            history = get_history(user_id)
            messages = build_messages(history)

            reply = await pollinations_text(session, messages, user_id)
            add_message(user_id, "assistant", reply)
            
            # Edit the "Thinking..." message to become the final response
            try:
                await thinking_msg.edit_text(reply, parse_mode=ParseMode.MARKDOWN)
            except Exception as md_error:
                logger.warning(f"Markdown parsing failed, editing as plain text: {md_error}")
                await thinking_msg.edit_text(reply)

        except Exception as e:
            logger.error(f"Handler error: {e}")
            await update.message.reply_text("Sorry, I encountered an unexpected error. Please try again.")


async def handle_image_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image messages with automatic analysis when caption is provided using openai-large model"""
    user_id = update.effective_user.id
    
    if update.message.caption:
        # If image has caption, analyze it automatically
        prompt = update.message.caption.strip()
        add_message(user_id, "user", f"Image with caption: {prompt}")
        
        # Send "Analyzing Image..." message
        analyzing_msg = await update.message.reply_text("Analyzing image...")
        
        try:
            # Get the highest quality photo
            photo = update.message.photo[-1]
            file = await photo.get_file()
            
            # Download image
            image_bytes = await file.download_as_bytearray()
            
            async with aiohttp.ClientSession() as session:
                # Analyze image with vision using openai-large model
                analysis = await analyze_image_with_vision(session, image_bytes, prompt, user_id)
                
                # Edit the analyzing message with the result
                try:
                    await analyzing_msg.edit_text(
                        f"**Image Analysis:**\n\n{analysis}",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as md_error:
                    logger.warning(f"Markdown parsing failed, editing as plain text: {md_error}")
                    await analyzing_msg.edit_text(f"Image Analysis: \n\n{analysis}")
                
                add_message(user_id, "assistant", f"[analyzed image: {analysis}]")
                
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            await analyzing_msg.edit_text("Sorry, I couldn't analyze that image. Please try again.")
    
    else:
        # If no caption, provide instructions
        await update.message.reply_text(
            "I see you sent an image! You can:\n"
            "• Add a caption to automatically analyze it with Ai l\n"
            "• Reply to this image with `/analyze <your question>`\n"
            "• Or just send it without analysis",
            parse_mode=ParseMode.MARKDOWN,
            reply_to_message_id=update.message.message_id
        )


# ---------- MAIN ----------
def main():
    try:
        app = Application.builder().token(TG_TOKEN).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("ClearHistory", clear_history_command))
        app.add_handler(CommandHandler("image", image_command))
        app.add_handler(CommandHandler("search", search_command))
        app.add_handler(CommandHandler("analyze", analyze_command))
        
        # Add message handlers
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_handler(MessageHandler(filters.PHOTO, handle_image_message))
        app.add_handler(MessageHandler(filters.Document.ALL, process_document))
        
        logger.info("DeepEx bot started successfully.")
        logger.info("Powered by DF-T4 | Developed by ELLIOTPXP")
        logger.info("Using openai-large model for text and vision")
        logger.info("Using standard image.pollinations.ai for image generation")
        logger.info("Document file support enabled")
        logger.info(f"Using OpenAI-compatible endpoint: {TEXT_URL}?token={TEXT_TOKEN}")
        app.run_polling()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")


if __name__ == "__main__":
    main()
