import json
import os

# Config
CHUNK_SIZE = 100  # words per chunk

def chunk_sentences(sentences, words, chunk_size=CHUNK_SIZE):
    """
    Split sentences into RAG-ready chunks with approx chunk_size words each.
    """
    chunks = []
    current_chunk = {"sentences": [], "words": []}
    word_count = 0
    chunk_id = 1

    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) > chunk_size and current_chunk["sentences"]:
            # Save current chunk
            current_chunk["chunk_id"] = chunk_id
            current_chunk["entities"] = {}  # Placeholder for NER
            current_chunk["summary"] = current_chunk["sentences"][0]
            chunks.append(current_chunk)

            # Start new chunk
            chunk_id += 1
            current_chunk = {"sentences": [], "words": []}
            word_count = 0

        current_chunk["sentences"].append(sentence)
        current_chunk["words"].extend(sentence_words)
        word_count += len(sentence_words)

    # Append last chunk if exists
    if current_chunk["sentences"]:
        current_chunk["chunk_id"] = chunk_id
        current_chunk["entities"] = {}
        current_chunk["summary"] = current_chunk["sentences"][0]
        chunks.append(current_chunk)

    return chunks

def chunking(stage2_output: dict, doc_id: str = "unknown", output_file=None):
    """
    Update the Stage 2 dict with chunks (Stage 3 info).
    """
    sentences = stage2_output.get("sentences", [])
    words = stage2_output.get("words", [])

    # Create chunks
    chunks = chunk_sentences(sentences, words, CHUNK_SIZE)

    # Update original dict instead of creating a new one
    stage2_output.update({
        "doc_id": doc_id,
        "chunks": chunks
    })

    # Save JSON if path provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stage2_output, f, indent=2, ensure_ascii=False)
        print(f"🎯 Stage 3 results saved to {output_file}")

    return stage2_output

# if __name__ == "__main__":
#     # Example Stage 2 string (like the one you showed)
#     stage2_str = '''
#     {
#       "cleaned_text": "sample html kmrl project this is a sample html file for testing extraction. feature 1: classification feature 2: summarization",
#       "sentences": [
#         "sample html kmrl project this is a sample html file for testing extraction.",
#         "feature 1: classification feature 2: summarization"
#       ],
#       "words": [
#         "sample", "html", "kmrl", "project", "sample", "html", "file", "testing",
#         "extraction", "feature", "<NUM>", "classification", "feature", "<NUM>", "summarization"
#       ],
#       "num_sentences": 2,
#       "num_words": 15
#     }
#     '''
#     stage2_output = json.loads(stage2_str)
#     chunking(stage2_output, "../processed_output/stage3_results.json")
