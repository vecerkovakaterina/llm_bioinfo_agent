import requests
import json

from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION
from langchain.globals import set_debug, set_verbose


set_debug(True)
set_verbose(True)


def call_ollama_api(params):
    ollama_api_generate = "http://localhost:11434/api/chat"
    request = requests.post(ollama_api_generate, json=params)
    return request


def get_response_answer(ollama_api_generate_response):
    return ollama_api_generate_response.text


tools = [
    {
        "name": "look_up_species_by_ensembl_id",
        "description": "Look up the species for any gene Ensembl ID e.g. ENSG00000157764",
        "parameters": {
            "type": "object",
            "properties": {
                "ensembl_id": {
                    "type": "string",
                    "description": "Ensembl identifier of a gene",
                },
            },
            "required": ["ensembl_id"],
        },
    },
    {
        "name": "get_taxonomy_classification",
        "description": "Get taxonomy classification based on species name e.g. Homo sapiens",
        "parameters": {
            "type": "object",
            "properties": {
                "species_name": {
                    "type": "string",
                    "description": "Scientific name of a species",
                },
            },
            "required": ["species_name"],
        },
    },
    DEFAULT_RESPONSE_FUNCTION,
]

DEFAULT_SYSTEM_TEMPLATE = f"""You have access to the following tools:

{json.dumps(tools, indent=2)}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""

llm_call = get_response_answer(
    call_ollama_api(
        {
            "model": "llama3:70b-instruct",  # locally running mixtral:8x22b llama3:70b-instruct
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
                {
                    "role": "user",
                    "content": "What species has the gene ENSMUSG00000017167?",
                },
                {
                    "role": "assistant",
                    "content": '{\n"tool": "look_up_species_by_ensembl_id",\n"tool_input": {\n"ensembl_id": "ENSMUSG00000017167"\n}\n}',
                },
                {
                    "role": "user",
                    "content": "Function call look_up_species_by_ensembl_id with query ENSMUSG00000017167 result is Mus musculus",
                },
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
    )
)

print(llm_call)

# TODO logging
# TODO gget functions as tools
