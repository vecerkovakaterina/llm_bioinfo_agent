from langchain.globals import set_debug, set_verbose
from tools_vllm import *
import os
from dotenv import load_dotenv
from fireworks.client import Fireworks

load_dotenv("api_keys.env")

fireworks_api_key = os.getenv("FIREWORKS_API_KEY")


set_debug(True)
set_verbose(True)
# gget.setup("alphafold") keeps failing
# gget.setup("cellxgene")
# gget.setup("elm")


def call_fireworks_api(messages):
    client = Fireworks(
        api_key=fireworks_api_key, base_url="https://api.fireworks.ai/inference/v1"
    )

    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        messages=messages,
        temperature=0.0,
    )
    return chat_completion


def get_response_answer(fireworks_api_completion):
    return fireworks_api_completion.choices[0]


def parse_content(fireworks_api_completion):
    return fireworks_api_completion.message.content


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
    "query_cellxgene": query_cellxgene,
    "search_cosmic_for_mutations": search_cosmic_for_mutations,
    "align_sequences_with_diamond": align_sequences_with_diamond,
    "predict_eukaryotic_motif": predict_eukaryotic_motif,
    "perform_enrichment_analysis": perform_enrichment_analysis,
    "align_sequences_with_muscle": align_sequences_with_muscle,
    "mutate_sequences": mutate_sequences,
    "query_pdb": query_pdb,
    "get_fpt_link_to_reference_genome_by_species": get_fpt_link_to_reference_genome_by_species,
    "get_sequences_for_ensembl_ids": get_sequences_for_ensembl_ids,
}


# question = "What can you tell me about this gene: ENSMUSG00000050530"
question = "Is the species with the gene ENSMUSG00000050530 same as species with the gene ENSG00000139618?"
# question = "Can you please find the top 5 correlated genes to ACE2 in human?"
# question = "Please predict structure of protein with this sequence MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
# question = input("Enter your question: ")

response = None

messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
    {
        "role": "user",
        "content": question,
    },
]


# json.loads(chat_completion.choices[0].message.content)["tool"]

while response is None:

    print(f"Chatting with the model")
    llm_call = get_response_answer(call_fireworks_api(messages))
    print(llm_call)

    response = parse_response(llm_call)
    if not response:
        content = parse_content(llm_call)
        function_call, function_call_args = parse_function_call(llm_call)
        print(f"Calling function {function_call} with arguments {function_call_args}")
        messages.append({"role": "assistant", "content": content})
        if function_call in tools_dict:
            function_call_result = tools_dict[function_call](**function_call_args)
            print(f"Function call result {function_call_result}")
            messages.append(
                {
                    "role": "user",
                    "content": f"Function call {function_call} with query {function_call_args} is {function_call_result}",
                }
            )
        else:
            print(f"Tool {function_call} does not exist.")
            messages.append(
                {"role": "user", "content": f"Tool {function_call} does not exist."}
            )


print(response)

# TODO rename functions to be more descriptive
# TODO logging
# TODO after response wait for another human message
