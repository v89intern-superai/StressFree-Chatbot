# -*- coding: utf-8 -*-
"""
Standalone Model Debugging Script

สคริปต์นี้ใช้สำหรับทดสอบการโหลดและการทำงานของโมเดล AI โดยตรง
เพื่อแยกปัญหาที่เกี่ยวกับโมเดลออกจากปัญหาของ Docker หรือ FastAPI

วิธีใช้:
1. เปิด Terminal และเข้าไปที่ Directory 'backend/'
2. (ครั้งแรก) ติดตั้งไลบรารี: pip install -r requirements.txt
3. ตั้งค่า Hugging Face Token: export HUGGING_FACE_TOKEN='hf_...' (สำหรับ Mac/Linux)
   หรือ $env:HUGGING_FACE_TOKEN='hf_...' (สำหรับ PowerShell)
4. รันสคริปต์: python debug_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys
from peft import PeftModel, PeftConfig

# ==============================================================================
# 1. Configuration
# ==============================================================================
# อ่านค่า Model Path จาก Environment Variable, ถ้าไม่มีให้ใช้ค่าเริ่มต้น
MODEL_PATH = os.getenv("MODEL_PATH", "Pathfinder9362/Student-Mind-Mate-AI-v2")
OFFLOAD_DIR = "./offload_debug"
os.makedirs(OFFLOAD_DIR, exist_ok=True)


# ==============================================================================
# 2. ฟังก์ชันสำหรับโหลดโมเดล (Boilerplate)
# ==============================================================================
def load_peft_model(peft_model_id):
    """
    โหลดโมเดล LoRA ด้วยวิธีที่ประหยัดหน่วยความจำ
    """
    print(f"--- ⚙️  Starting Model Loading Process ---")
    print(f"🔄 Attempting to load model from: {peft_model_id}")

    # ตรวจสอบ Hugging Face Token
    if os.getenv("HUGGING_FACE_TOKEN") is None:
        print("⚠️  Warning: HUGGING_FACE_TOKEN environment variable not set.")
        print("   - This may cause issues if the model is private.")

    try:
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model_name = config.base_model_name_or_path
        print(f"✅ Found Base Model: {base_model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_use_double_quant=True
        )

        print("🔄 Loading Base Model in 4-bit...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=OFFLOAD_DIR,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        print("🔄 Loading Adapter (LoRA)...")
        model = PeftModel.from_pretrained(base_model, peft_model_id)

        print("✅ Model and Tokenizer loaded successfully!")
        print("------------------------------------------")
        return model, tokenizer
    except Exception as e:
        print(f"❌ CRITICAL ERROR during model loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==============================================================================
# 3. ฟังก์ชันสำหรับสร้างคำตอบ (Inference)
# ==============================================================================
def generate_response(prompt, model, tokenizer):
    """
    สร้างคำตอบจากโมเดล
    """
    system_prompt = (
        "คุณคือ 'เพื่อนใจวัยเรียน AI' เป็น AI Chatbot ที่มีความเข้าอกเข้าใจ, ให้กำลังใจ, และไม่ตัดสิน"
    )
    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    print("🤖 Model is generating a response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    new_tokens = outputs[0, input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


# ==============================================================================
# 4. Main Execution Loop
# ==============================================================================
if __name__ == "__main__":
    model, tokenizer = load_peft_model(MODEL_PATH)

    if model and tokenizer:
        print("\n🎉 Model is ready! Starting interactive chat session.")
        print("   - Type your message and press Enter.")
        print("   - Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                user_prompt = input("You: ")
                if user_prompt.lower() in ["quit", "exit"]:
                    print("👋 Exiting chat session. Goodbye!")
                    break
                
                response = generate_response(user_prompt, model, tokenizer)
                print(f"AI: {response}\n")

            except KeyboardInterrupt:
                print("\n👋 Exiting chat session. Goodbye!")
                break
            except Exception as e:
                print(f"🔥 An error occurred during inference: {e}")
    else:
        print("\nCould not start chat session because the model failed to load.")
