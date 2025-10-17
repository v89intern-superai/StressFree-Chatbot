 # -*- coding: utf-8 -*-
"""
โค้ดสำหรับ Fine-tune โมเดล OpenThaiGPT-1.5-7b
เพื่อสร้าง Chatbot 'เพื่อนใจวัยเรียน AI' (Student's Mind Mate AI)
สำหรับรันบน Google Colab

(เวอร์ชันปรับปรุงประสิทธิภาพขั้นสูงและจัดการหน่วยความจำ)

วิธีใช้:
1. ตั้งค่า Runtime ใน Colab เป็น T4 GPU
2. อัปโหลดไฟล์ 'dataset.jsonl'
3. รันเซลล์นี้ สคริปต์จะขอเชื่อมต่อ Google Drive และ Hugging Face
4. โมเดลและผลลัพธ์จะถูกบันทึกใน Google Drive ของคุณ
"""
# ==============================================================================
# 2. Import Libraries และตั้งค่าเริ่มต้น (Imports & Initial Setup)
# ==============================================================================
import torch
import gc
import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from huggingface_hub import notebook_login
from google.colab import drive
from google.colab import files
import sys # Import sys module

# --- ตรวจสอบว่ารันใน Google Colab หรือไม่ ---
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# 3. ฟังก์ชันและ Configuration หลัก (Core Functions & Config)
# ==============================================================================

def clear_gpu_memory():
    """ฟังก์ชันสำหรับเคลียร์ GPU memory และ RAM เพื่อป้องกัน OOM errors"""
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Memory cleared.")

print("🚀 Thai Student Mind Mate AI Fine-Tuning Setup Started...")

# --- เชื่อมต่อ Google Drive ---
print("📁 กำลังเชื่อมต่อ Google Drive...")
try:
    drive.mount('/content/drive')
    output_dir = "/content/drive/MyDrive/SS5_SuperAi/finetune/finetune2209"
    print(f"✅ Google Drive connected. Output will be saved to: {output_dir}")
except Exception as e:
    output_dir = "./Student-Mind-Mate-AI/finetune-results"
    print(f"⚠️ ไม่สามารถเชื่อมต่อ Google Drive: {e}. Output will be saved locally to: {output_dir}")

os.makedirs(output_dir, exist_ok=True)
offload_dir = os.path.join(output_dir, "offload")
os.makedirs(offload_dir, exist_ok=True)

# --- เคลียร์หน่วยความจำก่อนเริ่ม ---
clear_gpu_memory()

# --- กำหนดค่า Configuration หลัก ---
print("⚙️ กำลังตั้งค่าคอนฟิกูชัน...")
model_name = "openthaigpt/openthaigpt1.5-7b-instruct"
new_model_name = "Student-Mind-Mate-AI-v2"

# **เพิ่ม:** กำหนด System Prompt ไว้ในโค้ดที่เดียว
SYSTEM_PROMPT = (
    "คุณคือ AI Chatbot ที่มีความเข้าอกเข้าใจ, ให้กำลังใจ, และไม่ตัดสิน โดยมีหน้าที่หลักดังนี้:\n"
    "1. เป็นเพื่อนรับฟังปัญหาและความเครียดของนักเรียน/นักศึกษา\n"
    "2. หากเป็นเรื่องการเรียน ให้ช่วยให้คำแนะนำในการวางแผนการเรียน การจัดการเวลา หรือเทคนิคที่ช่วยให้เรียนได้ดีขึ้น\n"
    "3. **สำคัญที่สุด:** หากผู้ใช้แสดงความเสี่ยงเกี่ยวกับการทำร้ายร่างกายตัวเอง ให้หยุดการให้คำแนะนำทั่วไปและแนะนำให้ติดต่อผู้เชี่ยวชาญทันที"
)

# --- ตั้งค่า Quantization (BitsAndBytesConfig) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage_dtype=torch.bfloat16,
)

# --- ตั้งค่า PEFT (LoRA) ---
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=4,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# --- ตั้งค่า Training Arguments ---
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
    eval_strategy="steps", # **แก้ไข:** เปลี่ยนจาก evaluation_strategy เป็น eval_strategy
    eval_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    save_total_limit=5,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none", # ปิดการ report ไปยัง services ภภภภภภภภภภภภภภภภภภภภภภภายนอก
)
print("✅ ตั้งค่าเสร็จสิ้น!")


def main():
    """ฟังก์ชันหลักสำหรับรันกระบวนการทั้งหมด"""
    try:
        # ==============================================================================
        # 4. โหลดและเตรียมข้อมูล (Data Loading & Preparation)
        # ==============================================================================
        print("\n📊 กำลังโหลดและเตรียมข้อมูล...")

        train_dataset = load_dataset("json", data_files="train_dataset.jsonl", split="train")
        eval_dataset = load_dataset("json", data_files="test_dataset.jsonl", split="train")
        print(f"✅ ข้อมูลถูกโหลด: {len(train_dataset)} train samples และ {len(eval_dataset)} eval samples")

        # **เพิ่ม:** ฟังก์ชันสำหรับเพิ่ม System Prompt
        def add_system_prompt(example):
            example["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            return example

        # **เพิ่ม:** ใช้ .map() เพื่อเพิ่ม system prompt ลงในทุกตัวอย่าง
        train_dataset = train_dataset.map(add_system_prompt)
        eval_dataset = eval_dataset.map(add_system_prompt)
        print("✅ System prompt ถูกเพิ่มลงในทุกตัวอย่างข้อมูลเรียบร้อยแล้ว")

        # ==============================================================================
        # 5. โหลดโมเดลและ Tokenizer (Model & Tokenizer Loading)
        # ==============================================================================
        print("\n🤖 กำลังโหลดโมเดล OpenThaiGPT และ Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = 256


        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir,
        )
        model.config.use_cache = False

        # --- เตรียมโมเดลสำหรับ K-bit training ---
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        print("\n trainable params:")
        model.print_trainable_parameters()
        print("✅ โมเดลและ LoRA ถูกตั้งค่าเรียบร้อย!")

        # ==============================================================================
        # 6. สร้างและเริ่ม Fine-tuning (Trainer Setup & Training)
        # ==============================================================================
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_arguments,
        )

        clear_gpu_memory()

        print("\n======================================")
        print("🚀      เริ่มต้นการ Fine-tuning...     ")
        print("======================================\n")

        trainer.train()

        print("\n======================================")
        print("🎉      Fine-tuning เสร็จสิ้น!         ")
        print("======================================\n")

        # --- บันทึกโมเดลสุดท้าย ---
        final_model_path = os.path.join(output_dir, "final_model")
        print(f"💾 กำลังบันทึกโมเดลสุดท้ายไปที่: {final_model_path}")
        trainer.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)


        # ==============================================================================
        # 7. อัปโหลดโมเดลขึ้น Hugging Face Hub (Upload)
        # ==============================================================================
        print("\n==========================================================")
        print("☁️      ขั้นตอนการอัปโหลดโมเดลขึ้น Hugging Face Hub      ")
        print("==========================================================")
        notebook_login()

        hf_username = "Pathfinder9362"
        hf_repo_id = f"{hf_username}/{new_model_name}"

        print(f"\n🚀 กำลังอัปโหลดโมเดลไปที่: {hf_repo_id} ...")
        trainer.model.push_to_hub(hf_repo_id, commit_message="End of training")
        tokenizer.push_to_hub(hf_repo_id, commit_message="End of training")
        print(f"\n🎉 อัปโหลดโมเดลสำเร็จ! ที่ https://huggingface.co/{hf_repo_id}")

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดร้ายแรงในกระบวนการหลัก: {e}")
        import traceback
        traceback.print_exc()
        try:
            emergency_path = os.path.join(output_dir, "checkpoint_emergency")
            trainer.save_model(emergency_path)
            print(f"💾 Emergency checkpoint saved to {emergency_path}")
        except:
            print("❌ ไม่สามารถบันทึก emergency checkpoint ได้")

    finally:
        print("\nกระบวนการทั้งหมดเสร็จสิ้น กำลังเคลียร์หน่วยความจำ...")
        clear_gpu_memory()


if __name__ == "__main__":
    main()
