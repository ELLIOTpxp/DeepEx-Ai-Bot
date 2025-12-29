#!/usr/bin/env python3
"""
Telegram bot with deep thinking, enhanced search, and rotating API keys.
"""

import asyncio
import re
import aiohttp
import logging
import itertools
import random
import json
import base64
import os
import tempfile
import io
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from telegram.error import BadRequest
from PIL import Image

# Configuration
TG_TOKEN = "8334089670:AAH7AbTMw9VRtPpgvxbaQqSOo_glkFB9mJE"

# Google Custom Search Keys
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

# Pollinations AI Keys
POLLINATIONS_KEYS = [
    "plln_sk_u2A56jiRkPJcOIvMyiTlBDMzhpOFhbSr",
    "plln_sk_4qTJHM5CJqMY0kWYY9t1nMWIjibcdROz",
    "plln_sk_hJHsBiKQE9rNhcXtO6BTj0UKNRroQfqN"
]

MAX_HISTORY_LENGTH = 1000
SYSTEM_PROMPT = ( 
        "You are KeRa, Deep Researcher, Multi-Layer Thinker. All-In-One Ai assistant that exist on Telegram. Created by TEAM FSY.\n\n"
        "Available Commands that can only use by user:\n"
        "• Normal text message will use for usual conversation\n"
        "• Sending an image with with captions will view by Ai and analyze.\n"
        "• /search <query> - Double-layer Deep web search\n"
        "• /deepthink <query> - Deep research & reasoning (Multi-Layer Research, reasoning and thinking)\n"
        "• /analyze <prompt> - Analyze replied image\n"
        "• /ClearHistory - Reset chat memory\n\n"
        "TEAM FSY | DEVELOPER CHANNEL : https://t.me/DeepExAi"
)
MAX_RETRIES = 5
TEXT_URL = "https://gen.pollinations.ai/v1/chat/completions"
IMAGE_URL = "https://gen.pollinations.ai/image/{prompt}"

# Logging setup
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Cycle through keys
google_key_cycle = itertools.cycle(GOOGLE_API_KEYS)
pollinations_key_cycle = itertools.cycle(POLLINATIONS_KEYS)

# Memory management
chat_memory = {}

def add_message(user_id: int, role: str, content: str):
    chat_memory.setdefault(user_id, [])
    chat_memory[user_id].append({"role": role, "content": content})
    if len(chat_memory[user_id]) > MAX_HISTORY_LENGTH:
        chat_memory[user_id] = chat_memory[user_id][-MAX_HISTORY_LENGTH:]

def get_history(user_id: int):
    return chat_memory.get(user_id, [])

def clear_history(user_id: int):
    if user_id in chat_memory:
        chat_memory[user_id] = []
        return True
    return False

def build_messages(history, system_prompt=SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system_prompt}]
    recent_history = history[-6:]
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    return messages

def build_messages_with_image(history, image_data: str, user_prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent_history = history[-4:]
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
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

# Text Safe Utilities
def escape_markdown(text: str) -> str:
    """
    Escapes special characters for MarkdownV2.
    """
    if not text:
        return ""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

async def safe_reply_text(update: Update, text: str, parse_mode=ParseMode.MARKDOWN):
    """
    Attempts to send text with formatting. If it fails, sends plain text.
    """
    try:
        await update.message.reply_text(text, parse_mode=parse_mode)
    except BadRequest:
        # Fallback to plain text if markdown parsing fails
        await update.message.reply_text(text, parse_mode=None)

async def safe_edit_text(message, text: str, parse_mode=ParseMode.MARKDOWN):
    """
    Attempts to edit message with formatting. If it fails, edits with plain text.
    """
    try:
        await message.edit_text(text, parse_mode=parse_mode)
    except BadRequest:
        await message.edit_text(text, parse_mode=None)

# API Interaction
async def fetch_pollinations_response(session: aiohttp.ClientSession, messages, model: str, seed: int = None):
    """
    Generic function to get response from Pollinations AI with Key Rotation
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "stream": False,
        "temperature": 0.7
    }
    if seed:
        payload["seed"] = seed

    # Retry logic with key rotation
    for attempt in range(MAX_RETRIES):
        # Rotate key on every attempt/request
        current_key = next(pollinations_key_cycle)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}"
        }

        try:
            async with session.post(
                TEXT_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['message']['content']
                        return content.strip()
                elif resp.status == 429:
                    logger.warning(f"Rate limited (429) on key ending ...{current_key[-4:]}, switching...")
                else:
                    logger.warning(f"API Error {resp.status} for model {model}")
        except Exception as e:
            logger.warning(f"Request failed for {model} (Attempt {attempt+1}): {e}")
            await asyncio.sleep(1)
    
    return None

async def pollinations_image(session: aiohttp.ClientSession, prompt: str):
    try:
        import aiohttp.helpers
        url = IMAGE_URL.format(prompt=aiohttp.helpers.quote(prompt)) + "?model=gptimage"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                return await resp.read()
        return None
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None

def apply_zoom(image_bytes: bytes, zoom_factor: float = 1.20):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        zoomed_image = image.resize((new_width, new_height), Image.LANCZOS)
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        cropped_image = zoomed_image.crop((left, top, right, bottom))
        output_buffer = io.BytesIO()
        cropped_image.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Zoom application error: {e}")
        return image_bytes

# Google Search Logic
async def fetch_webpage_content(url: str) -> str:
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    text = re.sub(r'<[^>]+>', ' ', html)
                    text = re.sub(r'\s+', ' ', text)
                    return text.strip()[:3000]
    except Exception:
        pass
    return ""

async def google_search(query: str):
    if not query.strip():
        return []
    
    for _ in range(len(GOOGLE_API_KEYS)):
        key = next(google_key_cycle)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={key}&cx={SEARCH_ENGINE_ID}&q={aiohttp.helpers.quote(query)}&num=5"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        return items
        except Exception as e:
            logger.error(f"Search error with key {key[:10]}...: {e}")
            continue
    return []

# File Processing
async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    
    file_name = document.file_name
    mime_type = document.mime_type
    
    processing_msg = await update.message.reply_text(f"Processing document: {file_name}...")
    
    try:
        file = await document.get_file()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
            temp_path = temp_file.name
        
        await file.download_to_drive(temp_path)
        
        with open(temp_path, 'rb') as f:
            file_content = f.read()
        
        os.unlink(temp_path)
        
        if mime_type and 'image' in mime_type:
            prompt = f"Analyze this image file named {file_name}. Describe what you see in detail."
            image_data = base64.b64encode(file_content).decode('utf-8')
            
            async with aiohttp.ClientSession() as session:
                messages = build_messages_with_image([], image_data, prompt)
                analysis = await fetch_pollinations_response(session, messages, "openai-large")
            
            if analysis:
                await safe_edit_text(processing_msg, f"**Document Analysis:**\n\n{analysis}")
                add_message(user_id, "user", f"[uploaded image document: {file_name}]")
                add_message(user_id, "assistant", analysis)
            else:
                await safe_edit_text(processing_msg, "Failed to analyze the image document.")
        else:
            await safe_edit_text(processing_msg, "Currently, only image documents are supported for deep analysis. Text processing coming soon.")
            
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        await safe_edit_text(processing_msg, "Error processing document.")

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "Welcome to KeRa!\n\n"
        "Multi-Layer Thinker. Multi-Layer Deep Researcher. All-In-One Ai assistant.\n\n" 
        "Model Powered by Pollinations.ai\n\n"
        "KeRa Developed by TEAM FSY\n\n"
        "Available Commands:\n"
        "• Normal text message will use for usual conversation.\n"
        "• Sending an image with with captions will view by Ai and analyze.\n"
        "• /search <query> - Double-layer Deep web search\n"
        "• /deepthink <query> - Deep research & reasoning (Multi-Layer Research, reasoning and thinking)\n"
        "• /analyze <prompt> - Analyze replied image\n"
        "• /ClearHistory - Reset chat memory\n\n"
        "TEAM FSY | DEVELOPER CHANNEL : https://t.me/DeepExAi"
    )
    await update.message.reply_text(welcome_text)

async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if clear_history(user_id):
        await update.message.reply_text("Chat history cleared.")
    else:
        await update.message.reply_text("History is already empty.")

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("Usage: /image <prompt>")
        return
    
    prompt = " ".join(context.args).strip()
    add_message(user_id, "user", f"/image {prompt}")
    
    msg = await update.message.reply_text("Generating image...")
    
    async with aiohttp.ClientSession() as session:
        img_bytes = await pollinations_image(session, prompt)
        
        if img_bytes:
            zoomed = apply_zoom(img_bytes)
            await msg.delete()
            await update.message.reply_photo(zoomed, caption=f"Generated: {prompt}")
            add_message(user_id, "assistant", "[sent image]")
        else:
            await msg.edit_text("Image generation failed.")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Reply to an image with /analyze")
        return
    
    prompt = " ".join(context.args).strip() or "Describe this image in detail."
    add_message(user_id, "user", f"/analyze {prompt}")
    
    msg = await update.message.reply_text("Analyzing...")
    
    try:
        photo = update.message.reply_to_message.photo[-1]
        file = await photo.get_file()
        img_bytes = await file.download_as_bytearray()
        img_data = base64.b64encode(img_bytes).decode('utf-8')
        
        async with aiohttp.ClientSession() as session:
            messages = build_messages_with_image(get_history(user_id), img_data, prompt)
            response = await fetch_pollinations_response(session, messages, "openai-large")
            
            if response:
                await safe_edit_text(msg, f"**Analysis:**\n\n{response}")
                add_message(user_id, "assistant", response)
            else:
                await msg.edit_text("Analysis failed.")
                
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        await msg.edit_text("Error analyzing image.")

# Enhanced Search Command
async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("Usage: /search <query>")
        return
    
    query = " ".join(context.args).strip()
    add_message(user_id, "user", f"/search {query}")
    
    status_msg = await update.message.reply_text(f"Searching web for: {query}...")
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Google Search
        google_results = await google_search(query)
        
        # Step 2: AI Web Search (Gemini-Search)
        search_prompt = f"Search the web for: {query}. Provide detailed, up-to-date information found."
        gemini_search_resp = await fetch_pollinations_response(
            session, 
            [{"role": "user", "content": search_prompt}], 
            "gemini-search"
        )
        
        # Prepare context for final synthesis
        context_text = f"User Query: {query}\n\nGoogle Results:\n"
        for item in google_results[:3]:
            context_text += f"- {item.get('title')}: {item.get('snippet')} ({item.get('link')})\n"
        
        if gemini_search_resp:
            context_text += f"\nAI Search Agent Findings:\n{gemini_search_resp}\n"
        
        # Step 3: Synthesis (OpenAI-Large)
        await status_msg.edit_text("Synthesizing information...")
        
        final_prompt = (
            "You are a helpful research assistant. Combine the following search results "
            "into a coherent, well-formatted answer. Cite sources where possible.\n\n"
            f"{context_text}"
        )
        
        final_response = await fetch_pollinations_response(
            session,
            [{"role": "user", "content": final_prompt}],
            "openai-large"
        )
        
        if final_response:
            await safe_edit_text(status_msg, final_response)
            add_message(user_id, "assistant", final_response)
        else:
            await status_msg.edit_text("Could not generate a search summary.")

# Deep Think Command
async def deepthink_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("Usage: /deepthink <complex query>")
        return
    
    query = " ".join(context.args).strip()
    add_message(user_id, "user", f"/deepthink {query}")
    
    status_msg = await update.message.reply_text("Deep Thinking... (Initializing)")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Step 1: Google Search for context
            await status_msg.edit_text("Deep Thinking... (Gathering Data)")
            google_results = await google_search(query)
            google_context = "\n".join([f"- {i.get('snippet')}" for i in google_results[:4]])
            
            # Step 2: Gemini Large (Initial Analysis)
            await status_msg.edit_text("Deep Thinking... (Rechecking details)")
            step1_prompt = f"Analyze this query based on these search snippets: {query}\nContext: {google_context}"
            step1_response = await fetch_pollinations_response(
                session,
                [{"role": "user", "content": step1_prompt}],
                "gemini-large"
            ) or "No initial analysis generated."
            
            # Step 3: Perplexity Reasoning (Critique and Improve)
            await status_msg.edit_text("Deep Thinking... (Deep Reasoning)")
            step2_prompt = (
                f"Original Query: {query}\n"
                f"Initial Analysis: {step1_response}\n\n"
                "Critique this analysis. Think deeply, find flaws, check facts, and provide a improved logic path."
            )
            step2_response = await fetch_pollinations_response(
                session,
                [{"role": "user", "content": step2_prompt}],
                "perplexity-reasoning"
            ) or step1_response
            
            # Step 4: OpenAI Large (Final Polish)
            await status_msg.edit_text("Deep Thinking... (Finalizing)")
            final_prompt = (
                f"User Query: {query}\n"
                f"Reasoning Process: {step2_response}\n\n"
                "Based on the reasoning above, write a comprehensive, detailed, and perfectly formatted final response."
            )
            final_response = await fetch_pollinations_response(
                session,
                [{"role": "user", "content": final_prompt}],
                "openai-large"
            )
            
            if final_response:
                await safe_edit_text(status_msg, final_response)
                add_message(user_id, "assistant", final_response)
            else:
                await status_msg.edit_text("Deep thinking process failed at the final stage.")
                
        except Exception as e:
            logger.error(f"Deepthink error: {e}")
            await status_msg.edit_text("An error occurred during the deep thinking process.")

# Standard Text Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if not text:
        return

    add_message(user_id, "user", text)
    
    # Send placeholder
    reply_msg = await update.message.reply_text("Thinking...")
    
    async with aiohttp.ClientSession() as session:
        history = get_history(user_id)
        messages = build_messages(history)
        
        # Using standard model for chat
        response = await fetch_pollinations_response(session, messages, "openai")
        
        if response:
            await safe_edit_text(reply_msg, response)
            add_message(user_id, "assistant", response)
        else:
            await reply_msg.edit_text("Sorry, I couldn't respond right now.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if update.message.caption:
        prompt = update.message.caption
        add_message(user_id, "user", f"Image with caption: {prompt}")
        
        msg = await update.message.reply_text("Analyzing image...")
        
        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()
            img_bytes = await file.download_as_bytearray()
            img_data = base64.b64encode(img_bytes).decode('utf-8')
            
            async with aiohttp.ClientSession() as session:
                messages = build_messages_with_image(get_history(user_id), img_data, prompt)
                response = await fetch_pollinations_response(session, messages, "openai-large")
                
                if response:
                    await safe_edit_text(msg, f"**Analysis:**\n\n{response}")
                    add_message(user_id, "assistant", response)
                else:
                    await msg.edit_text("Analysis failed.")
        except Exception as e:
            logger.error(f"Photo handler error: {e}")
            await msg.edit_text("Error processing image.")
    else:
        await update.message.reply_text(
            "I see an image! Add a caption to analyze it, or reply with /analyze."
        )

def main():
    try:
        app = Application.builder().token(TG_TOKEN).build()
        
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("ClearHistory", clear_history_command))
        app.add_handler(CommandHandler("image", image_command))
        app.add_handler(CommandHandler("search", search_command))
        app.add_handler(CommandHandler("deepthink", deepthink_command))
        app.add_handler(CommandHandler("analyze", analyze_command))
        
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
        app.add_handler(MessageHandler(filters.Document.ALL, process_document))
        
        logger.info("KeRa bot started.")
        logger.info("Modes: Search, DeepThink, Image Gen, Vision")
        
        app.run_polling()
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    main()
