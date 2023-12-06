import json
import os

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
#openai.organization = ""





def generate_summaries(chat, dialogues):
	with open("test_b_pos_1_prompts.json") as f:
		few_shot_pos_7 = json.load(f)
	generated_section_text_list = []
	for idx, dialogue in enumerate(dialogues):
		print("Processing {} sample".format(idx + 1))
		try:
			example_prompt = PromptTemplate(input_variables=["dialogue", "summary"],
											template="Dialogue:\n{dialogue}\n\nSummary:\n{summary}")
			prompt = FewShotPromptTemplate(examples=few_shot_pos_7[idx], example_prompt=example_prompt,
										   suffix="Dialogue: {input}\n\nSummary:\n", input_variables=["input"])
			human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
			chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
			generations = chat.generate([chat_prompt.format_prompt(input=dialogue).messages]).generations
			generated_section_text = generations[0][0].text
			generated_section_text_list.append(generated_section_text)
		except Exception as E:
			print("#### Exception ####")
			print(E)
			print("####")
			continue
	return generated_section_text_list


import csv
import os

def run_task_b_summarization(input_csv_file_path):
    # Define the output file path based on the input file path
    output_csv_file_path = os.path.splitext(input_csv_file_path)[0] + '_summaries.csv'
    
    # Read dialogues and encounter_ids from the CSV file
    dialogues = []
    encounter_ids = []
    with open(input_csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dialogues.append(row['dialogue'])
            encounter_ids.append(row['encounter_id'])
    
    # Initialize the chat object with GPT-4
    chat = ChatOpenAI(model_name='gpt-4', temperature=0., max_tokens=800)
    
    # Generate summaries for the dialogues
    generated_section_text_list = generate_summaries(chat, dialogues)
    
    # Clean the generated summaries
    generated_section_text_list = [x.strip().replace('\n', ' ').replace('\r', '') for x in generated_section_text_list]
    
    # Write the encounter_id and summaries to a new CSV file
    with open(output_csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['encounter_id', 'summary'])
        for encounter_id, summary in zip(encounter_ids, generated_section_text_list):
            writer.writerow([encounter_id, summary])
    
    return output_csv_file_path

input_csv_file_path = 'taskB_testset4participants_inputConversations.csv'
output_file = run_task_b_summarization(input_csv_file_path)
