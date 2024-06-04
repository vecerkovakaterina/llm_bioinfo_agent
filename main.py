from langchain.globals import set_debug, set_verbose
import requests
from tools import *


set_debug(True)
set_verbose(True)
# gget.setup("alphafold") keeps failing
gget.setup("cellxgene")


def call_ollama_api(params):
    ollama_api_generate = "http://localhost:11434/api/chat"
    request = requests.post(ollama_api_generate, json=params)
    return request


def get_response_answer(ollama_api_generate_response):
    return ollama_api_generate_response.text


def parse_content(json_llm_call):
    json_llm_call = json.loads(json_llm_call)
    message_content = json_llm_call["message"]["content"]
    return message_content


def parse_response(json_llm_call):
    if "response" not in json_llm_call:
        return None
    message_content = parse_content(json_llm_call)
    nested_content = json.loads(message_content)
    response = nested_content["tool_input"]["response"]
    return response


def parse_function_call(json_llm_call):
    message_content = parse_content(json_llm_call)
    nested_content = json.loads(message_content)
    tool = nested_content["tool"]
    tool_input = nested_content["tool_input"]
    return tool, tool_input


tools_dict = {
    "get_further_clarification": get_further_clarification,
    "gget_info": gget_info,
    "gget_search": gget_search,
    "gget_alphafold": gget_alphafold,
    "gget_archs4": gget_archs4,
    "gget_blast": gget_blast,
    "gget_blat": gget_blat,
    "gget_cellxgene": gget_cellxgene,
    "gget_cosmic": gget_cosmic,
}

question = "Is the species with the gene ENSMUSG00000050530 same as species with the gene ENSMUSG00000017167?"  # question = input("Enter your question: ")
response = None

params_dict = {
    "model": "llama3:70b-instruct",  # locally running mixtral:8x22b llama3:70b-instruct
    "messages": [
        {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": question,
        },
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0, "num_ctx": 8000},
}

while response is None:

    print(f"Chatting with the model")
    llm_call = get_response_answer(call_ollama_api(params_dict))
    print(llm_call)

    response = parse_response(llm_call)
    if not response:
        content = parse_content(llm_call)
        function_call, function_call_args = parse_function_call(llm_call)
        print(f"Calling function {function_call} with arguments {function_call_args}")
        function_call_result = tools_dict[function_call](**function_call_args)
        print(f"Function call result {function_call_result}")
        params_dict["messages"].append({"role": "assistant", "content": content})
        params_dict["messages"].append(
            {
                "role": "user",
                "content": f"Function call {function_call} with query {function_call_args} is {function_call_result}",
            }
        )

print(response)


# TODO logging
# TODO gget functions as tools
# TODO after response wait for another human message
