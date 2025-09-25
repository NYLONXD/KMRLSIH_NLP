import pdfplumber
import docx
import pytesseract
from PIL import Image
import pandas as pd
from bs4 import BeautifulSoup
import email
import os
# import json

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                im = page.to_image(resolution=300).original
                page_text = pytesseract.image_to_string(im, lang="mal+eng")
            text += page_text or ""
    return text.strip()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def extract_text_from_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image, lang="mal+eng")

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n")

def extract_text_from_eml(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        msg = email.message_from_file(f)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="ignore"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            parts.append(payload.decode(errors="ignore"))
    return "\n".join(parts)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""
    file_type = "unknown"

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
        file_type = "pdf"
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        file_type = "docx"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        file_type = "txt"
    elif ext in [".jpg", ".jpeg", ".png", ".tiff"]:
        text = extract_text_from_image(file_path)
        file_type = ext[1:]
    elif ext == ".csv":
        text = extract_text_from_csv(file_path)
        file_type = "csv"
    elif ext in [".xls", ".xlsx"]:
        text = extract_text_from_excel(file_path)
        file_type = ext[1:]
    elif ext == ".html":
        text = extract_text_from_html(file_path)
        file_type = "html"
    elif ext == ".eml":
        text = extract_text_from_eml(file_path)
        file_type = "eml"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    result = {
        "doc_id": os.path.basename(file_path),
        "raw_text": text,
        "file_type": file_type,
        "length": len(text.split())
    }
    return result


# if __name__ == "__main__":
#     sample_files = [
#         "sample.pdf",
#         "sample.docx",
#         "sample.txt",
#         "sample.jpg",
#         "sample.png",
#         "sample.csv",
#         "sample.xlsx",
#         "sample.html",
#         "sample.eml"
#     ]
#     for file in sample_files:
#         file_path = os.path.join("../TestData", file)
#         if os.path.exists(file_path):
#             print(f"\n--- Extracting from {file} ---")
#             output = extract_text(file_path)
#             print(json.dumps(output, indent=2, ensure_ascii=False))
