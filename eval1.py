# ================== IMPORTS & CONFIGURATION ==================

import os
import json
import time
import sys
import csv
import uuid
import random
import streamlit as st
import asyncio
import aiohttp
import re
import pandas as pd
import numpy as np
import together
import google.generativeai as genai
import math
import requests
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from prompt import JUDGE_RUBRIC_PROMPT # custom rubric prompt for evaluation
load_dotenv() # Load environment variables from .env file

# ================== DATASET PREPARATION ==================

from datasets import load_dataset
ds = load_dataset("Dahoas/instruct-synthetic-prompt-responses")# Load synthetic prompt-response dataset
train_ds = ds["train"]
df = train_ds.to_pandas()# Convert HuggingFace dataset to pandas DataFrame
df.to_csv("input.csv", index=False)

# ================== API KEY CONFIGURATION ==================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")   
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")   
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(" GEMINI_API_KEY not found. Please set it in your environment or .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenRouter client for model inference
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ================== MODEL DEFINITIONS ==================

model_1 = "mistralai/mistral-7b-instruct:free"
model_2 = "mistralai/mistral-small-24b-instruct-2501:free"
model_3 = "google/gemma-3-27b-it:free"
model_4 = "mistralai/mistral-small-3.1-24b-instruct:free"
model_5 = "meta-llama/llama-4-maverick:free"
model_GROQ1 = "llama-3.1-8b-instant"
model_GROQ2 = "meta-llama/llama-4-maverick-17b-128e-instruct"
model_GROQ3 = "openai/gpt-oss-120b"

# ================== SCORING FUNCTIONS ==================

def score_with_qwen(prompt: str, response: str) -> dict:
    """
    Evaluate a prompt-response pair using Qwen model hosted on OpenRouter.
    Returns structured JSON scores based on evaluation rubric.
    """
    try:
        full_prompt = JUDGE_RUBRIC_PROMPT + "\n\n"
        full_prompt += f"Prompt: '''{prompt}'''\n\n"
        full_prompt += f"Agent response: '''{response}'''\n\n"
        full_prompt += "Output EXACT JSON now."

        completion = client.chat.completions.create(
            model=model_1,
            messages=[
                {"role": "system", "content": "You are an evaluator that strictly outputs JSON only."},
                {"role": "user", "content": full_prompt},
            ],
        )

        raw_text = completion.choices[0].message.content.strip()

        # Clean JSON (remove extra text around it)
        if "{" in raw_text:
            json_str = raw_text[raw_text.find("{"): raw_text.rfind("}") + 1]
        else:
            json_str = raw_text

        return json.loads(json_str)

    except Exception as e:
        return {"scores": {}, "total_score": 0, "error": str(e)}


# ==== SCORING FUNCTION ====
def score_with_gemini(prompt: str, response: str) -> dict:
    """
    Evaluate a prompt-response pair using Google Gemini API.
    Returns JSON-formatted evaluation scores.
    """
    try:
        full_prompt = JUDGE_RUBRIC_PROMPT + "\n\n"
        full_prompt += f"Prompt: '''{prompt}'''\n\n"
        full_prompt += f"Agent response: '''{response}'''\n\n"
        full_prompt += "Output EXACT JSON now."

        model = genai.GenerativeModel("gemini-2.5-flash")
        result = model.generate_content(full_prompt)

        # Extract raw text
        raw_text = ""
        if result.candidates and result.candidates[0].content.parts:
            raw_text = result.candidates[0].content.parts[0].text
        else:
            raw_text = result.text

        # Clean JSON
        json_str = raw_text.strip()
        if "{" in json_str:
            json_str = json_str[json_str.find("{"): json_str.rfind("}") + 1]

        return json.loads(json_str)
    except Exception as e:
        return {"scores": {}, "total_score": 0, "error": str(e)}
    
    
def score_with_groq(prompt: str, response: str) -> dict:
    """
    Evaluate a prompt-response pair using Groq API models.
    Returns JSON evaluation scores.
    """
    try:
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return {"scores": {}, "total_score": 0, "error": "Groq API key not configured"}

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}"}
        
        full_prompt = JUDGE_RUBRIC_PROMPT + "\n\n"
        full_prompt += f"Prompt: '''{prompt}'''\n\n"
        full_prompt += f"Agent response: '''{response}'''\n\n"
        full_prompt += "Output EXACT JSON now."

        payload = {
            "model": model_GROQ1,   # 🔑 replace with Groq model you want
            "messages": [
                {"role": "system", "content": "You are an evaluator that strictly outputs JSON only."},
                {"role": "user", "content": full_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        raw_text = data["choices"][0]["message"]["content"].strip()

        # Extract clean JSON
        if "{" in raw_text:
            json_str = raw_text[raw_text.find("{"): raw_text.rfind("}") + 1]
        else:
            json_str = raw_text

        return json.loads(json_str)

    except Exception as e:
        return {"scores": {}, "total_score": 0, "error": str(e)}    

# ================== CSV EVALUATION PIPELINE ==================

expla =[]
def evaluate_csv(input_csv: str, output_csv: str, batch_size: int = 10, total_rows: int = None, scorer=None):
    df = pd.read_csv(input_csv)
    
    # Limit number of rows to process
    if total_rows is None or total_rows > len(df):
        total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    
    # If file exists, resume from last completed batch
    start_index = 0
    if os.path.exists(output_csv):
        os.remove(output_csv)

    start_time = time.time()
    all_results =[]
    
    # Process rows in batches    
    for batch_start in range(start_index, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = df.iloc[batch_start:batch_end]

        results = []
        for i, row in batch.iterrows():
            prompt = row["prompt"]
            response = row["response"]

            judge = scorer(prompt, response) if scorer else {}  
            scores = judge.get("scores", {})
            explanations = judge.get("explanations", {}) 
            # total = judge.get("total_score", 0)
            print(judge)            

            # Collect structured results
            results.append({
                "Agent_id": f"Agent_{i+1}",
                "prompt_id": f"Prompt_{i+1}",
                "prompt": prompt,
                "response": response,
                "instruction_following": scores.get("instruction_following", 0),
                "hallucination": scores.get("hallucination", 0),
                "assumption_control": scores.get("assumption_control", 0),
                "coherence_accuracy": scores.get("coherence_accuracy", 0),
                "instruction_following_expl": explanations.get("instruction_following", ""),
                "hallucination_expl": explanations.get("hallucination", ""),
                "assumption_control_expl": explanations.get("assumption_control", ""),
                "coherence_accuracy_expl": explanations.get("coherence_accuracy", ""),
                "total_score": np.mean([
                    scores.get("instruction_following", 0),
                    scores.get("hallucination", 0),
                    scores.get("assumption_control", 0),
                    scores.get("coherence_accuracy", 0)
                ])
            })

        # Convert batch to DataFrame
        out_df = pd.DataFrame(results)
        all_results.append(out_df)

        # Append to CSV (no header after first batch)
        if batch_start == 0 and not os.path.exists(output_csv):
            out_df.to_csv(output_csv, index=False, mode="w")
        else:
            out_df.to_csv(output_csv, index=False, mode="a", header=False)

        print(f"Processed rows {batch_start+1}–{batch_end} / {total_rows}")

    elapsed = time.time() - start_time
    print(f"Finished! Total time: {elapsed/60:.2f} minutes. Output saved to {output_csv}")
    
    # Return concatenated results
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()
