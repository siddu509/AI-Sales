from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(dotenv_path)
load_dotenv(dotenv_path=dotenv_path)

df = pd.DataFrame()
# openai_api_key = os.getenv('OPENAI_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
print("printing \n", openai_api_key)
openai_api_key = "sk-proj-kTVJVY3TbLQzr3NCcfx2T3BlbkFJsmaCOYrWPxSP5xzZxIYq"
client = OpenAI(api_key = openai_api_key)


input = "Hello world"
model = "text-embedding-3-small"

def get_embedding(input, model):
   input = input.replace("\n", " ")
   return client.embeddings.create(input = input, model=model)
#    return client.embeddings.create(input = input, model=model).data[0].embedding


print("Output \n", get_embedding(input,model))