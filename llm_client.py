import json
import os
import atexit
import requests
import sys
from tqdm.auto import tqdm
from openai import OpenAI
from termcolor import colored
import time
from utils import read_yaml_file, remove_punctuation, batchify

def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60


def form_request(data, type, **kwargs):

    if "davinci" in type:
        request_data = {
            "prompt": data,
            "max_tokens": 1000,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
            "logprobs": None,
            "stop": None,
            **kwargs,
        }
    else:
        # assert isinstance(data, str)
        messages_list = []
        messages_list.append({"role": "user", "content": data})
        request_data = {
            "model": type,
            "messages": messages_list,
            "max_tokens": 1000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            **kwargs,
        }
    # print(request_data)
    return request_data

def my_llm_init(auth_file="../auth.yaml", llm_type='davinci', setting="default"):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    # try:
    #     openai.api_type = auth['api_type']
    #     openai.api_base = auth["api_base"]
    #     openai.api_version = auth["api_version"]
    # except:
    #     pass
    # openai.api_key = auth["api_key"]
    return auth


def llm_init(auth_file="../auth.yaml", llm_type='deepseek', setting="default"):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    try:
        client = OpenAI(
                    base_url=auth["api_base"],
                    api_key=auth["api_key"],
                )
    except:
        print("Error in initializing the client")
        pass
    return client


def api_call(model_name, data):
    client = llm_init(f"./auth.yaml", model_name)

    # a dictionary named llm_model that maps different LLM names (deepseek, gpt4, turbo, davinci) to their respective model names  

    llm_model = {
        "deepseek": "deepseek/deepseek-r1-distill-qwen-32b",
        "gpt4": "openai/gpt-4o-mini",
        "turbo": "openai/gpt-3.5-turbo",
        "mistral": "mistralai/mistral-7b-instruct:free",
        "llama": "meta-llama/llama-3-8b-instruct:free",
    }

    response = client.chat.completions.create(
        model=llm_model[model_name],
        messages=[
            {
                "role": "user",
                "content": data,
            }
        ],
        max_tokens= 1000,
        top_p= 0.95,
        frequency_penalty= 0,
        presence_penalty= 0,
    )

    return response.choices[0].message.content


def llm_query(data, client, type, task, **config):
    hypos = []
    response = None
    # model_name = "davinci" if "davinci" in type else "turbo"
    model_name = type
    client = llm_init(f"./auth.yaml", model_name)

    # batch
    if isinstance(data, list):
        batch_data = batchify(data, 20)
        for batch in tqdm(batch_data):
            retried = 0
            # request_data = form_request(batch, model_name, **config)
            if "davinci" in type:
                # print(request_data)
                while True:
                    try:
                        response = api_call(model_name, batch)
                        response = [r["text"] for r in response]
                        break
                    except Exception as e:
                        error = str(e)
                        print("here 1 retring...", error)
                        second = extract_seconds(error, retried)
                        retried = retried + 1
                        time.sleep(second)
            else:
                response = []
                for data in tqdm(batch):
                    # request_data = form_request(data, type, **config)
                    while True:
                        try:
                            result = api_call(model_name, data)
                            # print(result)
                            response.append(result)
                            break
                        except Exception as e:
                            error = str(e)
                            print("here 2 retring...", error)
                            second = extract_seconds(error, retried)
                            retried = retried + 1
                            time.sleep(second)

            # print(response)
            if task:
                results = [str(r).strip().split("\n\n")[0] for r in response]
            else:
                results = [str(r).strip() for r in response]
            # print(results)
            # results = [str(r['text']).strip() for r in response]
            # print(results)
            hypos.extend(results)
    else:
        retried = 0
        while True:
            try:
                print(type)
                result = ""
                if "turbo" in type or 'gpt4' in type:
                    # request_data = form_request(data, type, **config)
                    response = api_call(model_name, data)
                    result = response.strip()
                    break
                else:
                    response = api_call(model_name, data)
                    result = response.strip()
                    break
            except Exception as e:
                error = str(e)
                print("here 3 retring...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(second)
        if task:
            result = result.split("\n\n")[0]

        hypos = result
    return hypos


def paraphrase(sentence, client, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]

    else:
        resample_template = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    # print(resample_template)
    results = llm_query(resample_template, client, type, False, **kwargs)
    return results


def llm_cls(dataset, client=None, type=None, **config):
    hypos = []
    results = llm_query(dataset, client=client, type=type, task=True, **config)
    if isinstance(results, str):
        results = [results]
    hypos = [remove_punctuation(r.lower()) for r in results]

    return hypos



if __name__ == "__main__":
    llm_client = None
    llm_type = 'turbo'
    start = time.time()
    data =  ["""Q: Tom bought a skateboard for $ 9.46 , and spent $ 9.56 on marbles . Tom 
also spent $ 14.50 on shorts . In total , how much did Tom spend on toys ?                                                 
A: Let's think step by step. """]
    config = llm_init(auth_file="auth.yaml", llm_type=llm_type, setting="default")
    para = llm_query(
        data[0], client=llm_client, type=llm_type, task=False, temperature=0, **config
    )
    print(para)
    end = time.time()
    print(end - start)
