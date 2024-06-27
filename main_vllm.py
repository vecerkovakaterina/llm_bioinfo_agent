from openai import OpenAI
from langchain.globals import set_debug, set_verbose
import requests
from tools_vllm import *


set_debug(True)
set_verbose(True)
# gget.setup("alphafold") keeps failing
# gget.setup("cellxgene")
# gget.setup("elm")


def call_vllm_api(params):
    vllm_api_generate = "http://localhost:8000/v1/chat/completions"
    request = requests.post(vllm_api_generate, json=params)
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
# question = "Is the species with the gene ENSMUSG00000050530 same as species with the gene ENSG00000139618?"
# question = "Can you please find the top 5 correlated genes to ACE2 in human?"
# question = "Please predict structure of protein with this sequence MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEKGEHEQAAHHADTAYAHHKHAEEHAAQAAKHDAEHHAPKPH"
# question = input("Enter your question: ")
question = "Describe the format of Ensembl ID"

response = None

params_dict = {
    "model": "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",  # locally running TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ with vLLM
    "messages": [
        {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": question,
        },
    ],
}

llm_call = get_response_answer(call_vllm_api(params_dict))
print(llm_call)


# TODO rename functions to be more descriptive
# TODO logging
# TODO after response wait for another human message
