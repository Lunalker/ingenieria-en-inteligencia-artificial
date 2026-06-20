from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('GITHUB_TOKEN', '')

kb = json.loads(Path('data/knowledge_base.json').read_text())
texts = [f"{d['title']}\n{d['content']}" for d in kb]
metas = [{'topic': d['topic'], 'id': d['id']} for d in kb]
sp = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
cs, ms = [], []
for i, t in enumerate(texts):
    c = sp.split_text(t)
    cs.extend(c)
    ms.extend([metas[i]] * len(c))

emb = OpenAIEmbeddings(
    base_url='https://models.github.ai/inference',
    api_key=token,
    model='text-embedding-3-small'
)
vs = FAISS.from_texts(cs, emb, metadatas=ms)
vs.save_local('data/faiss_index')
print('INDEX OK')
