# pip install llama-parse
# # pip uninstall llama-index  # run this if upgrading from v0.9.x or older
# pip install -U llama-index --upgrade --no-cache-dir --force-reinstall
# pip install llama-parse
from llama_parse import LlamaParse
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(dotenv_path)
load_dotenv(dotenv_path=dotenv_path)

df = pd.DataFrame()
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key = openai_api_key)


input = "Hello world"
model = "text-embedding-3-small"

def get_embedding(input, model):
   input = input.replace("\n", " ")
#    return client.embeddings.create(input = input, model=model)
   return client.embeddings.create(input = input, model=model).data[0].embedding


# print("Output \n", get_embedding(input,model))

def pdf_parser(file):

    parser = LlamaParse(
        api_key="",  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        verbose=True,
    )

    documents = parser.load_data(file)
    print("printing /n",documents)
    return documents


# sync
document = pdf_parser("Problem_statements.pdf")[0].text
embed_doc = get_embedding(document,model)
print(embed_doc)

# sync batch
# documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])

# async
# documents = await parser.aload_data("./my_file.pdf")

# async batch
# documents = await parser.aload_data(["./my_file1.pdf", "./my_file2.pdf"])