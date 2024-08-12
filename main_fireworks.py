from langchain.globals import set_debug, set_verbose
from tools_vllm import *
import os
from dotenv import load_dotenv
from fireworks.client import Fireworks
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from tools_vllm import __conversational_response

load_dotenv("api_keys.env")

fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ")

set_debug(True)
set_verbose(True)


def call_fireworks_api(messages):
    client = Fireworks(
        api_key=fireworks_api_key, base_url="https://api.fireworks.ai/inference/v1"
    )

    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct",
        messages=messages,
        temperature=0.0,
    )
    return chat_completion


def get_response_answer(fireworks_api_completion):
    return fireworks_api_completion.choices[0]


def parse_content(fireworks_api_completion):
    return fireworks_api_completion.message.content


def parse_function_call(message_content):
    message_content = message_content.replace("<|python_tag|>", '')
    nested_content = json.loads(message_content)
    tool = nested_content["tool"]
    tool_input = nested_content["tool_input"]
    return tool, tool_input


def parse_response(message_content):
    if "<|python_tag|>" in message_content:
        return None
    else:
        return message_content


def get_tokens(text):
    chat_template_text = tokenizer.apply_chat_template(text, tokenize=False)
    return tokenizer.encode(chat_template_text)



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
    "__conversational_response": __conversational_response,
}

# testing and debugging questions
# question = "Predict the structure of this sequence: GFKQDIATIRGDLRTYAQDIFLAFLNKYPDERRYFKNYVGKSDQELKSMAKFGDHTEKVFNLMMEVADRATDCVPLASDANTLVQMKQHSSLTTGNFEKLFVALVEYMRASGQSFDSQSWDRFGKNLVSALSSAGMK"
# question = "What can you tell me about this gene ENSG00000139618"
# question = "Find genes related to diabetes"
# question = "Find sequences with BLAST similar to this one: GFKQDIATIRGDLRTYAQDIFLAFLNKYPDERRYFKNYVGKSDQELKSMAKFGDHTEKVFNLMMEVADRATDCVPLASDANTLVQMKQHSSLTTGNFEKLFVALVEYMRASGQSFDSQSWDRFGKNLVSALSSAGMK"
question = "Get number of tested samples, genes, mutations, fusions, etc. with 'ovary' as primary tissue site"
# question = "Perform enrichment analysis on these genes: PHF14 RBM3 MSL1 PHF21A"
# question = "Align these two sequences GTGAACGTGACACGTGCTCGAG and GGACAGTACTACGTGCAGTCAGTA"
# question = "Apply mutations from file test_mutations.csv to this sequence: GTGAACGTGACACGTGCTCGAG"
# question = "Find record 7S7U in PDB"
# question = "Get me the sequence of this gene ENSG00000139618"

response = None

messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
    {
        "role": "user",
        "content": question,
    },
]


while response is None:

    print(f"Chatting with the model")
    llm_call = get_response_answer(call_fireworks_api(messages))
    print(llm_call)

    content = parse_content(llm_call)
    response = parse_response(content)
    if not response:
        function_call, function_call_args = parse_function_call(content)
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
    elif response:
        print(response)
        response = None
        question = input()
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

# TODO rename functions to be more descriptive
# TODO logging
# TODO after response wait for another human message
