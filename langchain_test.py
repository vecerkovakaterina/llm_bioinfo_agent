import json

from langchain.globals import set_debug, set_verbose
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_experimental.llms.ollama_functions import (
    OllamaFunctions,
    convert_to_ollama_tool,
    DEFAULT_RESPONSE_FUNCTION,
)
import requests
from langchain_core.tools import tool

set_debug(True)
set_verbose(True)

ensembl_api = "https://rest.ensembl.org"


@tool
def look_up_species_by_ensembl_id(ensembl_id):
    """Look up the species for any gene Ensembl ID e.g. ENSG00000157764"""
    ext = f"/lookup/id/{ensembl_id}?"
    request = requests.get(
        ensembl_api + ext, headers={"Content-Type": "application/json"}
    )
    return request.json()["species"]


@tool
def get_taxonomy_classification(species_name):
    """Get taxonomy classification based on species name e.g. Homo sapiens"""
    ext = f"/taxonomy/classification/{species_name}?"
    request = requests.get(
        ensembl_api + ext, headers={"Content-Type": "application/json"}
    )
    return request.json()


tools = [
    look_up_species_by_ensembl_id,
    get_taxonomy_classification,
    DEFAULT_RESPONSE_FUNCTION,
]
llm = OllamaFunctions(model="llama3:70b-instruct", format="json", verbose=True)


llm = llm.bind_tools(
    tools=[
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
)

print(
    llm.invoke(
        [
            HumanMessage(content="What species has the gene ENSMUSG00000017167?"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "look_up_species_by_ensembl_id",
                        "arguments": '{"ensembl_id": "ENSMUSG00000017167"}',
                    }
                },
            ),
            HumanMessage(
                content="Results of look_up_species_by_ensembl_id with query `ENSMUSG00000017167`: Mus musculus. Describe this result in plain English"
            ),
        ]
    )
)
