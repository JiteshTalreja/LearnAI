from dotenv import load_dotenv



from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if "__main__" == __name__:
    # Load the text file
    loader = TextLoader("mediumblog.txt", autodetect_encoding=True)
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name="medium-embeddings",
    )