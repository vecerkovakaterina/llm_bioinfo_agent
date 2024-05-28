import json

import gget
from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION

tools = [
    {
        "name": "get_further_clarification",
        "description": "Ask user for missing information, further clarification, instructions or more details",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to the user",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "gget_info",
        "description": "Fetch extensive gene and metadata from Ensembl, UniProt and NCBI using Ensembl IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "ensembl_ids": {
                    "type": "list",
                    "description": "Ensembl IDs of genes to search",
                },
                "ncbi": {
                    "type": "bool",
                    "description": "Whether to include data from NCBI (default: True)",
                },
                "uniprot": {
                    "type": "bool",
                    "description": "Whether to include data from UniProt (default: True)",
                },
                "pdb": {
                    "type": "bool",
                    "description": "Whether to include data from PDB (default: False)",
                },
            },
            "required": ["ensembl_ids"],
        },
    },
    {
        "name": "gget_search",
        "description": "Fetch genes and transcripts from Ensembl using free-form search terms. Results are matched "
        "based on the 'gene name', 'description' and 'synonym' sections in the Ensembl database.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_words": {
                    "type": "list",
                    "description": "Free form search words (not case-sensitive) as a string or list of strings ("
                    "e.g.searchwords = ['GABA', 'gamma-aminobutyric']).",
                },
                "release": {
                    "type": "int",
                    "description": "Defines the Ensembl release number from which the files are fetched, e.g. 104. "
                    "This argument is overwritten if a specific database (which includes a release "
                    "number) is passed to the species argument. Default: None -> latest Ensembl "
                    "release is used",
                },
                "id_type": {
                    "type": "string",
                    "description": "'gene' (default) or 'transcript' Defines whether genes or transcripts matching "
                    "the searchwords are returned.",
                },
                "andor": {
                    "type": "string",
                    "description": "'or' (default) or 'and'. 'or': Returns all genes that INCLUDE AT LEAST ONE of the "
                    "searchwords in their name/description. 'and': Returns only genes that INCLUDE ALL "
                    "of the searchwords in their name/description.",
                },
            },
            "required": ["search_words"],
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


# tools implementations


def get_further_clarification(question):
    """Ask user for missing information further clarification, instructions or more details"""
    answer = input(question)
    return answer


def gget_info(ensembl_ids, ncbi=True, uniprot=True, pdb=False):
    """Fetch extensive gene and metadata from Ensembl, UniProt and NCBI using Ensembl IDs"""
    if not ensembl_ids:
        return "Required argument ensembl_ids not provided!"
    result = gget.info(
        ens_ids=ensembl_ids, ncbi=ncbi, uniprot=uniprot, pdb=pdb, verbose=False
    )
    if not result.empty:
        result = result.to_string()
    return result


def gget_search(search_words, release=None, id_type="gene", andor="or"):
    """Fetch genes and transcripts from Ensembl using free-form search terms.
    Results are matched based on the "gene name", "description" and "synonym" sections in the Ensembl database.
    """
    if not search_words:
        return "Required argument search_words not provided!"
    result = gget.search(
        searchwords=search_words,
        release=release,
        id_type=id_type,
        andor=andor,
        verbose=False,
    )
    if not result.empty:
        result = result.to_string()
    return result
