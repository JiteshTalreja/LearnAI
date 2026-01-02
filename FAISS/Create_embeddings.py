import faiss
from pypdf import PdfReader
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_sentences_from_pdf(file_path):
    """Extract sentences from a PDF file."""

    # This function is called for every text snippet found on the page
    full_book = ""
    with open(file_path, "rb") as file:
        text = PdfReader(file)

        for page in text.pages:
            content = page.extract_text()
            if content:
                full_book += content + "\n"


    return nltk.sent_tokenize(full_book)

sentences_to_transform = extract_sentences_from_pdf("the_hundred_page_language_models_book.pdf")

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences_to_transform)
print(f"Generated {len(sentence_embeddings)} embeddings with shape {sentence_embeddings.shape}")

# Save embeddings to numpy file
np.save("sentence_embeddings.npy", sentence_embeddings)
print("Embeddings saved to sentence_embeddings.npy")

# save sentences to a text file
with open("sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences_to_transform:
        f.write(sentence.replace("\n", " ") + "\n")
print("Sentences saved to sentences.txt")

# print(full_book)
