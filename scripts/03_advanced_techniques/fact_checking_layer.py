# -*- coding: utf-8 -*-
"""
Fact-Checking Layer Script (เวอร์ชันสำหรับ Google Colab)

สคริปต์นี้จะสาธิตวิธีการใช้ LLM ตัวเดิมมาทำหน้าที่เป็น "ผู้ตรวจสอบข้อเท็จจริง"
เพื่อลดปัญหา Hallucination และเพิ่มความน่าเชื่อถือของคำตอบ

(เวอร์ชันปรับปรุง: ลดความซ้ำซ้อนและเน้นที่ตรรกะหลัก)
"""

# ==============================================================================
# 1. ติดตั้งและ Import Libraries
# ==============================================================================
print("⏳ กำลังติดตั้งไลบรารีที่จำเป็น...")


import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys
from huggingface_hub import notebook_login
from peft import PeftModel, PeftConfig
print("✅ ติดตั้งไลบรารีเสร็จสิ้น!")
# ==============================================================================
# 2. Configuration
# ==============================================================================
MODEL_PATH = "Pathfinder9362/Student-Mind-Mate-AI-v2"
OFFLOAD_DIR = "./offload_fact_check"
os.makedirs(OFFLOAD_DIR, exist_ok=True)
IN_COLAB = 'google.colab' in sys.modules

# ==============================================================================
# 3. (BOILERPLATE) ฟังก์ชันสำหรับโหลดโมเดล
# หมายเหตุ: ส่วนนี้เป็นโค้ดมาตรฐานที่คล้ายกับสคริปต์ QA ก่อนหน้า
# ==============================================================================
def load_peft_model(peft_model_id):
    """โหลดโมเดล LoRA ด้วยวิธีที่ประหยัดหน่วยความจำ"""
    print(f"\n🔄 กำลังโหลดโมเดลจาก: {peft_model_id}...")
    try:
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model_name = config.base_model_name_or_path

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, offload_folder=OFFLOAD_DIR, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(base_model, peft_model_id)
        print("✅ โหลดโมเดลสำเร็จ!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, None

# ==============================================================================
# 4. (CORE LOGIC) ฟังก์ชันสำหรับ Fact-Checking Layer
# ==============================================================================

def get_initial_answer(prompt, model, tokenizer):
    """รอบที่ 1: สร้างคำตอบในฐานะ 'เพื่อนใจวัยเรียน AI'"""
    system_prompt = (
       "คุณคือ 'เพื่อนใจวัยเรียน AI' เป็น AI Chatbot ที่มีความเข้าอกเข้าใจ, ให้กำลังใจ, และไม่ตัดสิน โดยมีหน้าที่หลักดังนี้:\n"
        "1. เป็นเพื่อนรับฟังปัญหาและความเครียดของนักเรียน/นักศึกษา\n"
        "2. หากเป็นเรื่องการเรียน ให้ช่วยให้คำแนะนำในการวางแผนการเรียน การจัดการเวลา หรือเทคนิคที่ช่วยให้เรียนได้ดีขึ้น\n"
        "3. **สำคัญที่สุด:** หากผู้ใช้แสดงความเสี่ยงเกี่ยวกับการทำร้ายร่างกายตัวเองหรือมีความคิดที่อันตราย ให้หยุดการให้คำแนะนำทั่วไปและแนะนำให้ติดต่อผู้เชี่ยวชาญทันที โดยให้ข้อมูลเบอร์โทรศัพท์ เช่น สายด่วนสุขภาพจิต 1323 หรือแนะนำให้ปรึกษาอาจารย์ที่ปรึกษาหรือนักจิตวิทยา"
    )
    chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

def get_fact_check_verdict(prompt, answer, model, tokenizer):
    """รอบที่ 2: ตรวจสอบคำตอบในฐานะ 'ผู้ตรวจสอบข้อเท็จจริง'"""
    system_prompt = (
        "คุณคือผู้ตรวจสอบข้อเท็จจริง AI ที่เชี่ยวชาญด้านสุขภาพจิต ภารกิจของคุณคือตรวจสอบความถูกต้องและความปลอดภัยของคำตอบที่ AI สร้างขึ้น\n"
        "โปรดวิเคราะห์คำตอบตามเกณฑ์ต่อไปนี้ และให้ผลลัพธ์ในรูปแบบ JSON เท่านั้น:\n"
        "1. **ความถูกต้องของข้อมูล (Factual Correctness):** เนื้อหาในคำตอบต้องเป็นข้อมูลที่ถูกต้องและไม่สร้างความเข้าใจผิด\n"
        "2. **ความปลอดภัย (Safety):** คำตอบจะต้องไม่มีเนื้อหาที่เป็นอันตราย, ยั่วยุ, หรือแนะนำสิ่งที่อาจส่งผลให้ผู้ใช้รู้สึกแย่ลงหรือเสี่ยงที่จะทำร้ายร่างกายตัวเองโดยเด็ดขาด\n\n"
        "**โครงสร้างผลลัพธ์ (JSON Output):**\n"
        "{\n"
        "  \"is_factually_correct\": boolean, // ตั้งค่าเป็น false หากไม่ผ่านเกณฑ์ข้อ 1 หรือ 2\n"
        "  \"explanation\": \"อธิบายเหตุผลสั้นๆ โดยระบุส่วนที่ถูกต้อง, ส่วนที่อาจเป็นข้อมูลเท็จ, หรือส่วนที่เป็นอันตราย\"\n"
        "}"
    )
    user_prompt = f"คำถามเดิม: \"{prompt}\"\n\nคำตอบที่ต้องตรวจสอบ: \"{answer}\""
    chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    json_str = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"is_factually_correct": False, "explanation": "รูปแบบผลลัพธ์ไม่ถูกต้อง"}

# ==============================================================================
# 5. (MAIN) ฟังก์ชันหลักในการรัน
# ==============================================================================
def checking():
    """รันกระบวนการทั้งหมด"""
    if IN_COLAB:
        notebook_login()

    model, tokenizer = load_peft_model(MODEL_PATH)
    if not model: return

    user_prompt = "ฉันจะเดินทางไปซื้อข้าวยังไง"

    print("\n" + "="*50 + "\n🚀      เริ่มต้นกระบวนการ Fact-Checking      🚀\n" + "="*50 + "\n")

    # --- 1. สร้างคำตอบเบื้องต้น ---
    print(f"👤 User Prompt: \"{user_prompt}\"\n\n--- 📝 ขั้นตอนที่ 1: กำลังสร้างคำตอบเบื้องต้น... ---")
    candidate_answer = get_initial_answer(user_prompt, model, tokenizer)
    print(f"🤖 คำตอบเบื้องต้น:\n\"{candidate_answer}\"")

    # --- 2. ทำการตรวจสอบ ---
    print("\n--- 🔍 ขั้นตอนที่ 2: กำลังส่งคำตอบไปตรวจสอบ... ---")
    verdict = get_fact_check_verdict(user_prompt, candidate_answer, model, tokenizer)
    print("📊 ผลการตรวจสอบ:")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    # --- 3. ตัดสินใจและสร้างคำตอบสุดท้าย ---
    print("\n--- ✅ ขั้นตอนที่ 3: การตัดสินใจสุดท้าย... ---")
    if verdict.get("is_factually_correct", False):
        final_answer = candidate_answer
        print("👍 ผลการตรวจสอบ: ผ่าน! คำตอบมีความน่าเชื่อถือสูง")
    else:
        final_answer = (
            "เราพบข้อมูลที่เป็นประโยชน์เกี่ยวกับเทคนิคการหายใจที่ได้รับการยอมรับ เช่น เทคนิค 4-7-8 ครับ "
            "อย่างไรก็ตาม สำหรับบางเทคนิคที่ถามมา เรายังไม่มีข้อมูลที่น่าเชื่อถือเพียงพอ จึงขออนุญาตไม่ให้ข้อมูลในส่วนนั้นเพื่อความถูกต้องนะครับ"
        )
        print(f"👎 ผลการตรวจสอบ: ไม่ผ่าน! เหตุผล: {verdict.get('explanation')}")
        print("   - กำลังสร้างคำตอบที่ปลอดภัย...")

    print("\n" + "="*50 + "\n💬      คำตอบสุดท้ายที่จะส่งให้ผู้ใช้      💬\n" + "="*50)
    print(f"\"{final_answer}\"")

if __name__ == "__main__":
    checking()
