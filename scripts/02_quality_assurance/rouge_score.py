# -*- coding: utf-8 -*-
"""
Recalculate ROUGE Score Script (เวอร์ชันสำหรับ Google Colab)

สคริปต์นี้ใช้สำหรับคำนวณคะแนน ROUGE จากไฟล์ผลลัพธ์ qa_results.csv
เพื่อประเมินความคล้ายคลึงระหว่างคำตอบที่ AI สร้างขึ้นกับคำตอบต้นแบบ

วิธีใช้ใน Google Colab:
1. รันเซลล์นี้
2. ระบบจะติดตั้งไลบรารีที่จำเป็น
3. ระบบจะแสดงปุ่ม "Choose Files" ให้คุณอัปโหลดไฟล์ `qa_results.csv`
4. สคริปต์จะคำนวณและแสดงคะแนน ROUGE ของแต่ละรายการ พร้อมสรุปค่าเฉลี่ย
"""

# ==============================================================================
# 1. ติดตั้งและ Import Libraries
# ==============================================================================
print("⏳ กำลังติดตั้งไลบรารีที่จำเป็น...")
# !pip install -q -U pythainlp
print("✅ ติดตั้งไลบรารีเสร็จสิ้น!")

import pandas as pd
import sys
import os
from pythainlp.tokenize import word_tokenize
from collections import Counter
from difflib import SequenceMatcher

# --- ตรวจสอบว่ารันใน Google Colab หรือไม่ ---
IN_COLAB = 'google.colab' in sys.modules

# ==============================================================================
# 2. Configuration & File Upload
# ==============================================================================
QA_RESULTS_FILE = None

if IN_COLAB:
    from google.colab import files
    print("\n📂 กรุณาอัปโหลดไฟล์ `qa_results.csv` ของคุณ:")
    uploaded = files.upload()

    if not uploaded:
        sys.exit("❌ ไม่พบไฟล์ที่อัปโหลด! กรุณารันเซลล์ใหม่อีกครั้ง")
    QA_RESULTS_FILE = next(iter(uploaded))
    print(f"✅ อัปโหลดไฟล์ `{QA_RESULTS_FILE}` สำเร็จ!")
else:
    # สำหรับการรันแบบ Local, ให้แน่ใจว่าไฟล์อยู่ใน Directory เดียวกัน
    QA_RESULTS_FILE = "qa_results.csv"
    if not os.path.exists(QA_RESULTS_FILE):
        sys.exit(f"❌ ไม่พบไฟล์ {QA_RESULTS_FILE} ใน Directory นี้!")

# ==============================================================================
# 3. (CORE LOGIC) ฟังก์ชัน ROUGE ที่สร้างขึ้นเอง
# ==============================================================================

def _get_ngrams(n, text_tokens):
    """สร้าง n-grams จากลิสต์ของ tokens"""
    return Counter(tuple(text_tokens[i:i+n]) for i in range(len(text_tokens) - n + 1))

def _calculate_f1(precision, recall):
    """คำนวณ F1-Score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_rouge_n(reference_tokens, generated_tokens, n):
    """คำนวณ ROUGE-N (N=1, 2, ...)"""
    ref_ngrams = _get_ngrams(n, reference_tokens)
    gen_ngrams = _get_ngrams(n, generated_tokens)

    overlapping_ngrams = ref_ngrams & gen_ngrams
    overlapping_count = sum(overlapping_ngrams.values())

    ref_total = sum(ref_ngrams.values())
    gen_total = sum(gen_ngrams.values())

    precision = overlapping_count / gen_total if gen_total > 0 else 0.0
    recall = overlapping_count / ref_total if ref_total > 0 else 0.0
    f1 = _calculate_f1(precision, recall)

    return f1

def calculate_rouge_l(reference_tokens, generated_tokens):
    """คำนวณ ROUGE-L (Longest Common Subsequence)"""
    matcher = SequenceMatcher(None, reference_tokens, generated_tokens)
    lcs_length = matcher.find_longest_match(0, len(reference_tokens), 0, len(generated_tokens)).size

    ref_total = len(reference_tokens)
    gen_total = len(generated_tokens)

    precision = lcs_length / gen_total if gen_total > 0 else 0.0
    recall = lcs_length / ref_total if ref_total > 0 else 0.0
    f1 = _calculate_f1(precision, recall)

    return f1


# ==============================================================================
# 4. ฟังก์ชันหลักในการคำนวณจากไฟล์ CSV
# ==============================================================================
def process_qa_results(file_path):
    """
    อ่านไฟล์ CSV, คำนวณคะแนน ROUGE ด้วยฟังก์ชันที่สร้างเอง, และแสดงผลสรุป
    """
    print(f"\n🔄 กำลังอ่านข้อมูลจากไฟล์: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if 'reference_answer' not in df.columns or 'generated_answer' not in df.columns:
            sys.exit("❌ ไฟล์ CSV ต้องมีคอลัมน์ 'reference_answer' และ 'generated_answer'")
    except Exception as e:
        sys.exit(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")

    results = []
    print("\n=============================================")
    print("📊      กำลังคำนวณคะแนน ROUGE...      📊")
    print("=============================================\n")

    for index, row in df.iterrows():
        reference = str(row['reference_answer'])
        generated = str(row['generated_answer'])
        case_id = row.get('id', index + 1)

        # ตัดคำภาษาไทยด้วย pythainlp
        ref_tokens = word_tokenize(reference, engine="newmm")
        gen_tokens = word_tokenize(generated, engine="newmm")

        # คำนวณคะแนนด้วยฟังก์ชันของเราเอง
        rouge1 = calculate_rouge_n(ref_tokens, gen_tokens, 1)
        rouge2 = calculate_rouge_n(ref_tokens, gen_tokens, 2)
        rougeL = calculate_rouge_l(ref_tokens, gen_tokens)

        processed_scores = {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL
        }
        results.append(processed_scores)

        print(f"--- เคส #{case_id} ---")
        print(f"  - ROUGE-1: {processed_scores['rouge1']:.4f}")
        print(f"  - ROUGE-2: {processed_scores['rouge2']:.4f}")
        print(f"  - ROUGE-L: {processed_scores['rougeL']:.4f}")
        print("-" * 20)



    # --- สรุปผลค่าเฉลี่ย ---
    if results:
        avg_df = pd.DataFrame(results)
        avg_scores = avg_df.mean()

        print("\n=============================================")
        print("📈         สรุปผลคะแนน ROUGE เฉลี่ย         📈")
        print("=============================================")
        print(f"  - Average ROUGE-1: {avg_scores['rouge1']:.4f}")
        print(f"  - Average ROUGE-2: {avg_scores['rouge2']:.4f}")
        print(f"  - Average ROUGE-L: {avg_scores['rougeL']:.4f}")
        print("=============================================\n")
    else:
        print("⚠️ ไม่สามารถคำนวณคะแนนได้ เนื่องจากไม่มีข้อมูลที่ถูกต้อง")


if __name__ == "__main__":
    process_qa_results(QA_RESULTS_FILE)
