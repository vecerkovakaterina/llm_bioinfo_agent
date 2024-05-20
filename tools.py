import json

import requests
from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION

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

ensembl_api = "https://rest.ensembl.org"


# tools implementations
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
    return request.json()[0]["parent"]["scientific_name"]
