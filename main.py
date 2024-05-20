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


ensembl_api = "https://rest.ensembl.org"


def look_up_species_by_ensembl_id(ensembl_id):
    """Look up the species for any gene Ensembl ID e.g. ENSG00000157764"""
    ext = f"/lookup/id/{ensembl_id}?"
    request = requests.get(
        ensembl_api + ext, headers={"Content-Type": "application/json"}
    )
    return request.json()["species"]


def get_taxonomy_classification(species_name):
    """Get taxonomy classification based on species name e.g. Homo sapiens"""
    ext = f"/taxonomy/classification/{species_name}?"
    request = requests.get(
        ensembl_api + ext, headers={"Content-Type": "application/json"}
    )
    return request.json()


tools_dict = {
    "look_up_species_by_ensembl_id": look_up_species_by_ensembl_id,
    "get_taxonomy_classification": get_taxonomy_classification,
}

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
                # {
                #     "role": "assistant",
                #     "content": '{\n"tool": "look_up_species_by_ensembl_id",\n"tool_input": {\n"ensembl_id": "ENSMUSG00000017167"\n}\n}',
                # },
                # {
                #     "role": "user",
                #     "content": "Function call look_up_species_by_ensembl_id with query ENSMUSG00000017167 result is Mus musculus",
                # },
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
    )
)


def parse_response(json_llm_call):
    if "response" not in json_llm_call:
        return None
    json_llm_call = json.loads(json_llm_call)
    message_content = json_llm_call["message"]["content"]
    nested_content = json.loads(message_content)
    response = nested_content["tool_input"]["response"]
    return response


def parse_function_call(json_llm_call):
    json_llm_call = json.loads(json_llm_call)
    message_content = json_llm_call["message"]["content"]
    nested_content = json.loads(message_content)
    tool = nested_content["tool"]
    tool_input = nested_content["tool_input"]
    return tool, tool_input


print(llm_call)
print(parse_response(llm_call))
function_call, function_args = parse_function_call(llm_call)

print(tools_dict[function_call](**function_args))

# TODO parse output for function calls
# TODO logging
# TODO gget functions as tools
# TODO tools to a separate file
