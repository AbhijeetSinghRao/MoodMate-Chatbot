import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
import random
import datetime
import os

# Load BlenderBot model
MODEL_NAME = "facebook/blenderbot-3B"

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

# Load emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# Constants
crisis_keywords = ["suicide", "self-harm", "kill myself", "hurt myself", "ending my life", "depressed"]
affirmations = [
    "You are doing better than you think.",
    "Your feelings are valid.",
    "This too shall pass.",
    "You are not alone in this.",
    "Small steps count too.",
    "You deserve peace and kindness.",
]
LOG_FILE = "chat_history.txt"

# Page config
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Support Chatbot")
st.markdown("I'm here for you. Let's talk. 💬")

# Mood
mood = st.radio("How are you feeling today?", ["😊 Happy", "😔 Sad", "😟 Anxious", "😌 Calm", "😤 Angry"])
if mood:
    st.success(f"It's okay to feel {mood.split()[1]}. Thanks for sharing.")

# Daily affirmation
st.subheader("🌞 Daily Affirmation")
st.info(random.choice(affirmations))

user_input = st.text_input("You:", placeholder="Type how you're feeling...")

# Session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Response logic
if user_input:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat.append(("You", user_input, timestamp))

    # Emotion detection
    emotion = emotion_classifier(user_input)[0]["label"]

    # Crisis detection
    if any(word in user_input.lower() for word in crisis_keywords):
        response = "⚠️ I'm really concerned about you. Please talk to someone you trust or contact a helpline. You are not alone. ❤️"
    else:
        inputs = tokenizer([user_input], return_tensors="pt")
        reply_ids = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Add bot response
    st.session_state.chat.append(("Bot", f"{response} (Detected emotion: *{emotion}*)", timestamp))

    # Save to file
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] You: {user_input}\n")
        f.write(f"[{timestamp}] Bot: {response}\n")

# Display history
st.subheader("🗨️ Chat History")
for speaker, msg, time in st.session_state.chat[::-1]:
    st.markdown(f"**{speaker}** ({time}): {msg}")
