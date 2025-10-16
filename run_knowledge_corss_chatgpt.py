import json
import logging
from time import sleep, localtime, strftime
from os import path
import jsonlines
import openai
from openai import OpenAI
from tqdm import tqdm
import os 
import re
import tiktoken  # 需要安装: pip install tiktoken

model = "gpt-3.5-turbo"

openai.api_key = "sk-p4XA7OFAfmfsmHhz8bCd99135f4b43A39b81Db97F8691f0e"

openai.base_url = "https://api.gpt.ge/v1/"
# openai.base_url = "https://api.vveai.com/v1/"
openai.default_headers = {"x-foo": "true"}



def modify_text(input_text,gemini_konwledge,deep_knowledge):
    # Split the input by 'Text:' to separate all the items
   
    head_item, example_item  = input_text.split('1 relation will be in the sentence.\n\n')

    
    head_item+="Now I am given you some example.\n"
    last_text_index = example_item .rfind("Text:")
    
    previous_reslut = "In addition, I provide some results obtained by other LLMs for reference. \n"
    previous_reslut+="LLMs 1:{0} .\n".format(gemini_konwledge)
    previous_reslut+="LLMs 2:{0} .\n\n".format(deep_knowledge)

    output_string = example_item [:last_text_index] + "\n==== USER ====\n" + previous_reslut+example_item [last_text_index:]
   
    out_put_item = head_item+output_string 


    
    return out_put_item 


def count_tokens(text, model_name="gpt-3.5-turbo"):
    """计算文本的令牌数量"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # 默认编码
    return len(encoding.encode(text))


def getNewContent(data):
    print(data)
    
    # 计算输入令牌
    input_tokens = count_tokens(data, model)
    print(f"输入令牌数: {input_tokens}")
    
    response = openai.chat.completions.create(
        model="{}".format(model),
        messages=[
            {"role": "user", "content": "{}".format(data)},
        ],
    )
    
    # 计算输出令牌
    output_content = response.choices[0].message.content
    output_tokens = count_tokens(output_content, model)
    total_tokens = input_tokens + output_tokens
    
    print(f"输出令牌数: {output_tokens}")
    print(f"总令牌数: {total_tokens}")
    
    return output_content, {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens, 
        'total_tokens': total_tokens
    }


def getPrompt(prompt, gemini, deep):

    reprompt  = modify_text(prompt,gemini,deep)

    return reprompt


def read_json_items(filepath):
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    return items


if __name__ == "__main__":

    seed = 17
    for type in ["test_V2"]:
        for topk in [8]:

                data = read_json_items('./{0}/{1}_to_answer_top{2}_cosine_similarity.json'.format(seed,type, topk))



                with open('./{0}/knowledge/{1}_{2}_{3}_cross.json'.format(seed,type,topk,"gemini-1.5-pro-002"), 'r', encoding='utf-8') as f:
                    cross_data_gemini = [json.loads(line) for line in f]
                with open('./{0}/knowledge/{1}_{2}_{3}_cross.json'.format(seed,type,topk,"deepseek-chat"), 'r', encoding='utf-8') as f:
                    cross_data_deep = [json.loads(line) for line in f]


                

                if not os.path.exists('./{0}/knowledge/{1}_{2}_{3}_cross_1.json'.format(seed,type,topk, model)):
                    open('./{0}/knowledge/{1}_{2}_{3}_cross_1.json'.format(seed,type,topk, model), 'x').close()
                new_f = open('./{0}/knowledge/{1}_{2}_{3}_cross_1.json'.format(seed,type,topk,model), 'a+',encoding='utf-8')

                with open('./{0}/knowledge/{1}_{2}_{3}_cross_1.json'.format(seed,type,topk,model), 'r', encoding='utf-8') as f:
                    print('./{0}/knowledge/{1}_{2}_{3}_cross_1.json'.format(seed,type,topk,model))
                    new_data = f.readlines()
                
                number = len(new_data)
         
                print("----------------number----------------")
                print(number)
                number = 0

                # filepath = './{0}/{1}.json'.format(seed,type)
                # with open(filepath, 'r', encoding='utf-8') as f:
                #     train_data = json.load(f)
                #     train_data_new = train_data[number:]

                filepath = './{0}/{1}.json'.format(seed,type)
                print(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    train_data_new = []
                    for line in f:
                        train_data_new.append(json.loads(line))
                    # print(len(train_data_new))
                    # exit()
                    train_data_new = train_data_new[number:]
                
                cross_data_gemini_new= cross_data_gemini[number:]
                cross_data_deep_new = cross_data_deep[number:]
                print(cross_data_deep_new)
                # for index, item in enumerate(data[number:]):
                #     prompt = item
                #     print(cross_data_deep_new[index])
                #     prompt = getPrompt(prompt, cross_data_gemini_new[index]['knowledge'], cross_data_deep_new[index]['knowledge'])
                #     new_content = getNewContent(prompt)
                #     print(new_content)
                #     train_data_new[index]['knowledge'] = new_content
                #     print(train_data_new[index])
                    # new_f.write(json.dumps(train_data_new[index]))
                    # new_f.write('\n')

 
                import time
                call_times = []  # 存储每次调用的时间
                token_stats = {
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_tokens': 0,
                    'samples': []
                }
                print(len(train_data_new))

                for index, item in enumerate(data[number:]):
                    prompt = item
                    print(f"正在处理第 {index + 1} 条样本...")
                    print(cross_data_deep_new[index])
                    
                    start_time = time.time()
                    prompt = getPrompt(prompt, cross_data_gemini_new[index]['knowledge'], cross_data_deep_new[index]['knowledge'])
                    new_content, tokens_info = getNewContent(prompt)
                    end_time = time.time()
                    
                    # 统计令牌
                    token_stats['total_input_tokens'] += tokens_info['input_tokens']
                    token_stats['total_output_tokens'] += tokens_info['output_tokens']
                    token_stats['total_tokens'] += tokens_info['total_tokens']
                    token_stats['samples'].append(tokens_info)
                    
                    call_time = end_time - start_time
                    call_times.append(call_time)
                    
                    print(new_content)
                    train_data_new[index]['knowledge'] = new_content
                    print(train_data_new[index])
                    
                    print(f"第 {index + 1} 条样本处理完成，耗时: {call_time:.2f} 秒")
                    
                    # 可选：限制处理数量进行测试
                    if index == 15:
                        break

                # 统计信息
                if call_times:
                    avg_time = sum(call_times) / len(call_times)
                    max_time = max(call_times)
                    min_time = min(call_times)
                    total_time = sum(call_times)
                    
                    # 令牌统计
                    avg_input_tokens = token_stats['total_input_tokens'] / len(token_stats['samples'])
                    avg_output_tokens = token_stats['total_output_tokens'] / len(token_stats['samples'])
                    avg_total_tokens = token_stats['total_tokens'] / len(token_stats['samples'])
                    
                    print(f"\n=== 处理统计 ===")
                    print(f"总样本数: {len(call_times)}")
                    print(f"总耗时: {total_time:.2f} 秒")
                    print(f"平均耗时: {avg_time:.2f} 秒")
                    print(f"最长耗时: {max_time:.2f} 秒")
                    print(f"最短耗时: {min_time:.2f} 秒")
                    
                    print(f"\n=== 令牌统计 ===")
                    print(f"总输入令牌: {token_stats['total_input_tokens']:,}")
                    print(f"总输出令牌: {token_stats['total_output_tokens']:,}")
                    print(f"总令牌数: {token_stats['total_tokens']:,}")
                    print(f"平均输入令牌/样本: {avg_input_tokens:.1f}")
                    print(f"平均输出令牌/样本: {avg_output_tokens:.1f}")
                    print(f"平均总令牌/样本: {avg_total_tokens:.1f}")


