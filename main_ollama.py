from langchain.globals import set_debug, set_verbose
import requests
from tools_ollama import *


set_debug(True)
set_verbose(True)
# gget.setup("alphafold") keeps failing
# gget.setup("cellxgene")
# gget.setup("elm")


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
    "get_info_for_ensembl_ids": get_info_for_ensembl_ids,
    "search_ensembl": search_ensembl,
    "get_protein_structure_prediction": get_protein_structure_prediction,
    "get_correlated_genes": get_correlated_genes,
    "get_similar_sequences_with_blast": get_similar_sequences_with_blast,
    "get_similar_sequences_with_blat": get_similar_sequences_with_blat,
    "gget_cellxgene": gget_cellxgene,
    "gget_cosmic": gget_cosmic,
    "gget_diamond": gget_diamond,
    "gget_elm": gget_elm,
    "gget_enrichr": gget_enrichr,
    "gget_muscle": gget_muscle,
    "gget_mutate": gget_mutate,
    "gget_pdb": gget_pdb,
    "get_fpt_link_to_reference_genome_by_species": get_fpt_link_to_reference_genome_by_species,
    "get_sequences_for_ensembl_ids": get_sequences_for_ensembl_ids,
}


# question = "What can you tell me about this gene: ENSMUSG00000050530"
question = "Is the species with the gene ENSMUSG00000050530 same as species with the gene ENSG00000139618?"
# question = "Can you please find the top 5 correlated genes to ACE2 in human?"
# question = "Please predict structure of protein with this sequence MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
# question = input("Enter your question: ")

response = None

params_dict = {
    "model": "llama3:70b-instruct",  # locally running mixtral:8x22b llama3:70b-instruct with ollama
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

    print("Chatting with the model")
    llm_call = get_response_answer(call_ollama_api(params_dict))
    print(llm_call)

    response = parse_response(llm_call)
    if not response:
        content = parse_content(llm_call)
        function_call, function_call_args = parse_function_call(llm_call)
        print(f"Calling function {function_call} with arguments {function_call_args}")
        params_dict["messages"].append({"role": "assistant", "content": content})
        if function_call in tools_dict:
            function_call_result = tools_dict[function_call](**function_call_args)
            print(f"Function call result {function_call_result}")
            params_dict["messages"].append(
                {
                    "role": "user",
                    "content": f"Function call {function_call} with query {function_call_args} is {function_call_result}",
                }
            )
        else:
            print(f"Tool {function_call} does not exist.")
            params_dict["messages"].append(
                {"role": "user", "content": f"Tool {function_call} does not exist."}
            )


print(response)

# TODO rename functions to be more descriptive
# TODO change order of tools
# TODO logging
# TODO after response wait for another human message
