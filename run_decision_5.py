import json
import os
import openai
import random

model = "gpt-4o-mini"

openai.api_key = "sk-JAGO7XKH4dhBsNEd5c6127584934459180827f6cA4E70cEe"

openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}



def read_json_lines(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def fuse_and_summarize_knowledge(knowledge_list):
    prompt = """
Task Description:
You will extract key information from multiple model-generated results, fuse them, and summarize to produce a final, concise, and accurate output. Each model's result may contain similar or complementary information, and your task is to identify commonalities, differences, and synthesize the most reasonable conclusion.

Input Format:
You will receive multiple model-generated results, each providing an answer to the same question. These answers may differ in some details but generally revolve around the same topic.

Output Requirements:
Fuse Information: Extract key information from each model's result, identifying common points and differences.
Summarize Conclusion: Based on the extracted information, generate a final, concise, and accurate conclusion. Ensure the conclusion covers the core points from all models and resolves any redundancy or contradictions—no more than 200 tokens.
Model Results:
 """
    knowledge_list = list(knowledge_list)
    random.shuffle(knowledge_list)
    for index, knowledge in enumerate(knowledge_list):
        prompt += f"\n LLMs_{index}: {knowledge}"
    # prompt+="\n\n System Output (final result) :"
    prompt+="\n \n System OutputNote: The output format as the LLMs result as follows: The relation between Lana (Person) and Kanye (Person) is peer. The relation between Lana (Person) and Azealia Banks (Person) is also peer. Reasoning: The text discusses Lana ending a situation with Kanye and Azealia Banks, suggesting they are all contemporaries within the same industry, likely music or entertainment. The term \"ending\" implies a professional interaction or competitive dynamic rather than a romantic relationship. This context reinforces that Lana, Kanye, and Azealia Banks are peers, operating within a similar professional sphere and potentially exhibiting rivalry or competition"
    print(prompt)
    response = openai.chat.completions.create(
        model="{}".format(model),  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": "{}".format(prompt)},
        ],
    )
    print(response.choices[0].message.content) 
    return response.choices[0].message.content

if __name__ == "__main__":
    seeds = [17,67,97]
    topK = 5
    for seed in seeds:
        for type in ["train","val"]:
            base_path = '/Users/yuanli/self_other/other_code/Thor_JMERE/feature_simularity/{0}/knowledge'.format(seed)

            files = [
                '{0}_{1}_gemini-1.5-pro-002_cross.json'.format(type,topK),
                '{0}_{1}_gemini-1.5-pro-002_directly.json'.format(type,topK),
                '{0}_{1}_deepseek-chat_directly.json'.format(type,topK),
                '{0}_{1}_gpt-3.5-turbo_directly.json'.format(type,topK),
                '{0}_{1}_deepseek-chat_cross.json'.format(type,topK),
                '{0}_{1}_gpt-3.5-turbo_cross.json'.format(type,topK),

            ]

            random.shuffle(files)


            all_knowledge = []
            output_filepath = os.path.join(base_path, '{0}_{1}_final_knowledge.json'.format(type,topK))
            if not os.path.exists(output_filepath):
                open(output_filepath, 'x').close()
            new_f = open(output_filepath, 'a+',encoding='utf-8')

            with open(output_filepath, 'r', encoding='utf-8') as f:
                new_data = f.readlines()
                number = len(new_data)
            print("----------------number----------------")
            print(number)



            # Read all files and collect knowledge by lines
            knowledge_by_lines = [[] for _ in range(len(files))]
            for i, file in enumerate(files):
                filepath = os.path.join(base_path, file)
                data = read_json_lines(filepath)
                for item in data[number:]:
                    knowledge_by_lines[i].append(item['knowledge'])
            
            train_filepath = './{0}/{1}.json'.format(seed,type)
            # 这里是读取对应的 train.json 文件,然后替换原有Knowledge 字段。 
            with open(train_filepath, 'r', encoding='utf-8') as f:
                        train_data = json.load(f)
                        train_data_new = train_data[number:]


            # filepath = './{0}/{1}.json'.format(seed,type)
            # with open(filepath, 'r', encoding='utf-8') as f:
            #     train_data_new = []
            #     for line in f:
            #         train_data_new.append(json.loads(line))
            #     train_data_new = train_data_new[number:]

            for index, knowledge_list in enumerate(zip(*knowledge_by_lines)):
                print(knowledge_list)
                print("********************"*10)
                final_summary = fuse_and_summarize_knowledge(knowledge_list)
            
        
                train_data_new[index]['knowledge'] = final_summary
                print(train_data_new[index])
                print("---------------------------------"*10)
                new_f.write(json.dumps(train_data_new[index]))
                new_f.write('\n')


