import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
import io

import nltk
nltk.download('punkt')

import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

class DataGeneratorModule:
    def __init__(
            self,
            model_id,
            model_name,
            dataset,
            dataset_dir,
            prompttype,
            category,
            temperature
    ):
        super().__init__()
        self.model_id = model_id
        self.model_name = model_name
        self.prompttype = prompttype
        self.category = category
        self.temperature = temperature
        self.dataset = dataset
        self.dataset_dir = dataset_dir

    def return_prompt(self,prompt_type, book_name, author, prompt_subcategory = None):
        prompt_dict = {
            'Length':{
                'Without':f'Write a complete novel similar to {book_name} by {author}.',
                'Min':f'Write a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.',
                'Exactly':f'Write a complete novel similar to {book_name} by {author}. The novel should be exactly 10000 words.',
                'Max':f'Write a complete novel similar to {book_name} by {author}. The novel should be at most 10000 words.'
            },
            'Similarity':{
                'SimilarStyle':f'Write a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.',
                'SameStyle':f'Write a complete novel as same as {book_name} by {author}. The novel should be at least 10000 words.',
                'SameBackground':f'Write a complete novel with same background in {book_name} by {author}. The novel should be at least 10000 words.',
                'SameCharacters':f'Write a complete novel with same characters in {book_name} by {author}. The novel should be at least 10000 words.',
            },
            'Identification':{
                'BookName':f'Write a complete novel similar to {book_name}. The novel should be at least 10000 words.',
                'BookName-AuthorName':f'Write a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.'
            },
            'Chapter':{
                'FirstAndLast':f'Write the first and last chapters of a novel similar to {book_name} by {author}. The novel should be at least 10000 words.',
                'All':f'Write a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.',
                'First':f'Write the first chapter of a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.',
                'First5':f'Write first five chapters of a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.'
            },
            'Default': f'Write a complete novel similar to {book_name} by {author}. The novel should be at least 10000 words.'
        }

        if prompt_subcategory is not None:
          if prompt_type == 'Temperature':
            return prompt_dict['Default']
          else:
            return prompt_dict[prompt_type][prompt_subcategory]
        else:
          return prompt_dict['Default']

    def generate_similar_novel(self, row, prompt_type, prompt_subcategory=None, temperature=0.2):
        prompt = self.return_prompt(prompt_type, row['BookTitle'], row['Name'], prompt_subcategory)

        prompt_length = len(nltk.word_tokenize(prompt))
        max_length = 4000

        # TODO: use the longest prompt length
        response_length = max_length - prompt_length

        response = openai.ChatCompletion.create(
            model=self.model_id,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature,
            max_tokens=response_length
        )

        row['ChatGPTText'] = response.choices[0].message.content
        row['PromptType'] = prompt_type
        row['PromptSubCategory'] = prompt_subcategory

        return row

    def get_original_novels(self):
        dataset = load_dataset(self.dataset, data_dir=self.dataset_dir)
        data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test']), pd.DataFrame(dataset['validation'])])

        author_names = ['Arthur Conan Doyle', 'Henry Rider Haggard', 'Jack London', 'Mark Twain', 'Wilkie Collins']
        data = data[data['Name'].isin(author_names)]

        return data

    def generate_novels(self):
        data = self.get_original_novels()

        data_new = data.apply(lambda x: self.row_op(x, self.prompttype, self.category, temperature=self.temperature), axis=1)
        self.save_to_hf(data_new, self.prompttype, self.category)

    def row_op(self, row, prompttype, category, temperature):
        # while True:
          # Code executed here
          row = self.generate_similar_novel(row, prompttype, category, temperature)
          # time.sleep(10)
          return row

    def save_to_hf(self, dataset, prompttype, category):
        # huggingface_repo = '/content/ChatGPT'
        # data_new.to_csv(f'{huggingface_repo}/dataset_{prompttype}_{category}_{self.model_name}.csv', index=False)
        # # commit_msg = f'add {prompttype}_{category} dataset'
        # TODO: remove this dummy keywors
        # TODO: set local saving too
        output_path = f'dataset_{prompttype}_{category}_{self.model_name}_dummy.csv'
        dataset_path = 'Authorship/Forged-GAN-BERT'
        s_buf = io.BytesIO()
        dataset.to_csv(s_buf)
        s_buf.seek(0)
        api = HfApi()

        api.upload_file(
            path_or_fileobj=s_buf,
            path_in_repo=output_path,
            repo_id=dataset_path,
            repo_type="dataset",
        )
