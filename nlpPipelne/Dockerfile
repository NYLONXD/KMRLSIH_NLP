FROM python:3.11-slim

WORKDIR /app

# 1️⃣ Install Tesseract and language packs
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-mal && \
    rm -rf /var/lib/apt/lists/*

# 2️⃣ Copy only requirement files first (so dependencies are cached)

COPY requirements.txt ./nlp-requirements.txt

# 3️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r nlp-requirements.txt

# 4️⃣ Copy the rest of the project
COPY . .

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

CMD ["python","ProcessPipeline.py"]
