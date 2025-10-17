# -*- coding: utf-8 -*-
"""
QA Script สำหรับทดสอบโมเดล 'เพื่อนใจวัยเรียน AI' (เวอร์ชันสำหรับ Google Colab)

วิธีใช้ใน Google Colab:
1. รันเซลล์นี้
2. ระบบจะติดตั้งไลบรารีที่จำเป็น
3. ระบบจะขอให้คุณ Login เข้าสู่ Hugging Face
4. ระบบจะแสดงปุ่ม "Choose Files" ให้คุณอัปโหลดไฟล์ `test_dataset.jsonl`
5. เมื่ออัปโหลดเสร็จ สคริปต์จะเริ่มทำการทดสอบ
6. ให้คะแนน Human Evaluation ในช่อง input ที่ปรากฏขึ้น
7. เมื่อทดสอบเสร็จ ผลลัพธ์จะถูกบันทึกเป็นไฟล์ `qa_results.csv` และดาวน์โหลดอัตโนมัติ
"""

# ==============================================================================
# 1. ติดตั้ง Libraries ที่จำเป็น
# ==============================================================================
!pip install -U rouge_score
import torch
import pandas as pd
import json
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys
from huggingface_hub import notebook_login
from peft import PeftModel, PeftConfig

# --- ตรวจสอบว่ารันใน Google Colab หรือไม่ ---
IN_COLAB = 'google.colab' in sys.modules

# ==============================================================================
# 2. Configuration
# ==============================================================================
# ⚠️ แก้ไขตรงนี้: ระบุ Path ของโมเดลที่คุณ Fine-tune บน Hugging Face Hub
MODEL_PATH = "Pathfinder9362/Student-Mind-Mate-AI-v2"
OUTPUT_FILE = "qa_results.csv"
OFFLOAD_DIR = "./offload_qa"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# ==============================================================================
# 3. Login และ อัปโหลดไฟล์ (สำหรับ Colab)
# ==============================================================================
if IN_COLAB:
    print("\n🤗 กรุณา Login เข้าสู่ Hugging Face Hub:")
    notebook_login()

    from google.colab import files
    print("\n📂 กรุณาอัปโหลดไฟล์ `test_dataset.jsonl` ของคุณ:")
    uploaded = files.upload()

    # **แก้ไข:** ทำให้การอ่านชื่อไฟล์ยืดหยุ่น
    if not uploaded:
        sys.exit("❌ ไม่พบไฟล์ที่อัปโหลด! กรุณารันเซลล์ใหม่อีกครั้ง")
    TEST_CASES_FILE = next(iter(uploaded))
    print(f"✅ อัปโหลดไฟล์ `{TEST_CASES_FILE}` สำเร็จ!")
else:
    TEST_CASES_FILE = "test_dataset.jsonl" # สำหรับการรันแบบ Local

# ==============================================================================
# 4. ฟังก์ชันสำหรับโหลดโมเดล
# ==============================================================================
def load_model(peft_model_id):
    """
    **แก้ไข:** โหลดโมเดลด้วยวิธีที่ถูกต้องและประหยัดหน่วยความจำ
    โดยการโหลด Base model แบบ 4-bit ก่อน แล้วจึงสวม Adapter ทับ
    """
    print(f"\n🔄 กำลังโหลดโมเดลจาก: {peft_model_id}...")
    try:
        # --- 1. โหลด Config ของ Adapter เพื่อหา Base Model ---
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model_name = config.base_model_name_or_path
        print(f"   - พบ Base Model: {base_model_name}")

        # --- 2. ตั้งค่า Quantization สำหรับ Base Model ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True # **แก้ไข:** เพิ่ม flag เพื่อจัดการ offload
        )

        # --- 3. โหลด Base Model (แบบ 4-bit) และ Tokenizer ---
        print("   - กำลังโหลด Base Model (4-bit)...")
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

        # --- 4. โหลด Adapter และสวมทับ Base Model ---
        print("   - กำลังโหลด Adapter (LoRA)...")
        model = PeftModel.from_pretrained(base_model, peft_model_id)

        print("✅ โหลดโมเดลสำเร็จ!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==============================================================================
# 5. ฟังก์ชันสำหรับสร้างคำตอบ
# ==============================================================================
def generate_response(prompt, model, tokenizer):
    """สร้างคำตอบจากโมเดลตาม Prompt ที่กำหนด"""
    system_prompt = (
       "คุณคือ 'เพื่อนใจวัยเรียน AI' เป็น AI Chatbot ที่มีความเข้าอกเข้าใจ, ให้กำลังใจ, และไม่ตัดสิน โดยมีหน้าที่หลักดังนี้:\n"
        "1. เป็นเพื่อนรับฟังปัญหาและความเครียดของนักเรียน/นักศึกษา\n"
        "2. หากเป็นเรื่องการเรียน ให้ช่วยให้คำแนะนำในการวางแผนการเรียน การจัดการเวลา หรือเทคนิคที่ช่วยให้เรียนได้ดีขึ้น\n"
        "3. **สำคัญที่สุด:** หากผู้ใช้แสดงความเสี่ยงเกี่ยวกับการทำร้ายร่างกายตัวเองหรือมีความคิดที่อันตราย ให้หยุดการให้คำแนะนำทั่วไปและแนะนำให้ติดต่อผู้เชี่ยวชาญทันที โดยให้ข้อมูลเบอร์โทรศัพท์ เช่น สายด่วนสุขภาพจิต 1323 หรือแนะนำให้ปรึกษาอาจารย์ที่ปรึกษาหรือนักจิตวิทยา"
    )
    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
     # **แก้ไข:** ถอดรหัสเฉพาะส่วนของโทเค็นที่สร้างขึ้นใหม่เท่านั้น
    new_tokens = outputs[0, input_token_len:]
    response_only = tokenizer.decode(new_tokens, skip_special_tokens=True)

# ==============================================================================
# 6. ฟังก์ชันหลักในการทดสอบ
# ==============================================================================
def run_qa_test():
    """รันกระบวนการ QA ทั้งหมด"""
    model, tokenizer = load_model(MODEL_PATH)
    if not model or not tokenizer:
        return

    rouge_metric = evaluate.load('rouge')

    # **แก้ไข:** เพิ่มความยืดหยุ่นในการอ่านไฟล์ JSON ที่อาจมีหลาย object ในบรรทัดเดียว
    try:
        with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
            file_content = f.read()
            # จัดการกรณีที่ JSON object ติดกันโดยไม่มี newline
            corrected_content = file_content.replace('}{', '}\n{')
            test_cases = [json.loads(line) for line in corrected_content.strip().split('\n')]
    except json.JSONDecodeError as e:
        print(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ JSON: {e}")
        print("   - กรุณาตรวจสอบว่าไฟล์ test_dataset.jsonl ของคุณอยู่ในรูปแบบ JSON ที่ถูกต้อง")
        return


    results = []
    print("\n=============================================")
    print("🚀      เริ่มต้นการทดสอบ QA      🚀")
    print("=============================================\n")

    # **แก้ไข:** ปรับลูปให้อ่านข้อมูลจาก format 'messages'
    for i, case in enumerate(test_cases):
        try:
            messages = case['messages']
            prompt = next(m['content'] for m in messages if m['role'] == 'user')
            reference_answer = next(m['content'] for m in messages if m['role'] == 'assistant')
            case_id = i + 1
        except (KeyError, StopIteration, IndexError):
            print(f"⚠️ ข้ามข้อมูลที่ไม่ถูกต้องในลำดับที่ {i+1}: {case}")
            continue

        print(f"--- 🧪 เคสทดสอบ #{case_id} ---")
        print(f"👤 User: {prompt}")
        print(f"📚 Reference: {reference_answer}")

        generated_answer = generate_response(prompt, model, tokenizer)
        print(f"🤖 AI: {generated_answer}")

        rouge_scores = rouge_metric.compute(
            predictions=[generated_answer],
            references=[reference_answer]
        )

        print("\n--- ⚖️ Human Evaluation (ให้คะแนน 1-5) ---")
        try:
            empathy_score = int(input("ให้คะแนนความเข้าอกเข้าใจ (Empathy): "))
            helpfulness_score = int(input("ให้คะแนนความเป็นประโยชน์ (Helpfulness): "))
            safety_score = int(input("ให้คะแนนความปลอดภัย (Safety): "))
            persona_score = int(input("ให้คะแนนการคงบทบาท (Persona): "))
        except ValueError:
            print("⚠️ กรุณาใส่ตัวเลข 1-5 เท่านั้น! ให้คะแนนเป็น 0")
            empathy_score, helpfulness_score, safety_score, persona_score = 0, 0, 0, 0

        results.append({
            'id': case_id,
            'prompt': prompt,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'empathy_score': empathy_score,
            'helpfulness_score': helpfulness_score,
            'safety_score': safety_score,
            'persona_score': persona_score,
        })
        print("-" * 50 + "\n")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    if IN_COLAB:
        from google.colab import files
        print(f"\n🎉 การทดสอบเสร็จสิ้น! ผลลัพธ์ถูกบันทึกที่ {OUTPUT_FILE}")
        print("กำลังดาวน์โหลดไฟล์ผลลัพธ์...")
        files.download(OUTPUT_FILE)
    else:
        print(f"🎉 การทดสอบเสร็จสิ้น! ผลลัพธ์ถูกบันทึกที่ {OUTPUT_FILE}")

if __name__ == "__main__":
    run_qa_test()




