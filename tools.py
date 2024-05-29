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
                    "description": "Ensembl IDs of genes to search. Examples of Ensembl IDs are ENSMUSG00000017167, "
                    "ENSG00000139618 or ENSDARG00000024771. An Ensembl stable ID consists of five "
                    "parts: ENS(species)(object type)(identifier). (version). The second part is a "
                    "three-letter species code. For human, there is no species code so IDs are in the "
                    "form ENS(object type)(identifier).",
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
        "description": "Fetch genes and transcripts from Ensembl using free-form search terms, for example gene name "
        "like Brca2 or disease names like osteoarthrititis. Results are matched"
        "based on the 'gene name', 'description' and 'synonym' sections in the Ensembl database.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_words": {
                    "type": "list",
                    "description": "Free form search words (not case-sensitive) as a string or list of strings ("
                    "e.g.searchwords = ['GABA', 'gamma-aminobutyric']).",
                },
                "species": {
                    "type": "string",
                    "description": "Species or database to be searched. A species can be passed in the format "
                    "'genus_species', e.g. 'homo_sapiens' or 'arabidopsis_thaliana'.",
                },
                "release": {
                    "type": "int",
                    "description": "Defines the Ensembl release number from which the files are fetched, e.g. 104. "
                    "This argument is overwritten if a specific database (which includes a release "
                    "number) is passed to the species argument. Default: 111"
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
            "required": ["search_words", "species"],
        },
    },
    {
        "name": "gget_alphafold",
        "description": "Predicts the structure of a protein using a slightly simplified version of AlphaFold v2.3.0 "
        "published in the AlphaFold Colab notebook",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence (str), a list of sequences",
                },
                # "out": {
                #     "type": "string",
                #     "description": "Path to folder to save prediction results in (str)."
                #     "Default: './[date_time]_gget_alphafold_prediction'",
                # },
                "multimer_for_monomer": {
                    "type": "bool",
                    "description": "Use multimer model for a monomer (default: False).",
                },
                "multimer_recycles": {
                    "type": "int",
                    "description": "The multimer model will continue recycling until the predictions stop changing, "
                    "up to the limit set here (default: 3). For higher accuracy, at the potential cost "
                    "of longer inference times, set this to 20.",
                },
                "relax": {
                    "type": "bool",
                    "description": "True/False whether to AMBER relax the best model (default: False).",
                },
                "plot": {
                    "type": "bool",
                    "description": "True/False whether to provide a graphical overview of the prediction (default: "
                    "True).",
                },
                "show_sidechains": {
                    "type": "bool",
                    "description": "True/False whether to show side chains in the plot (default: True).",
                },
            },
            "required": ["sequence"],
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

If you don't have all required arguments to use a tool, use the get_further_clarification tool to get missing 
information from the user."""


# tools implementations


def get_further_clarification(question):
    """Ask user for missing information further clarification, instructions or more details."""
    answer = input(question)
    return answer


def gget_info(ensembl_ids, ncbi=True, uniprot=True, pdb=False):
    """Fetch extensive gene and metadata from Ensembl, UniProt and NCBI using Ensembl IDs. Examples of Ensembl IDs
    are ENSMUSG00000017167, ENSG00000139618 or ENSDARG00000024771. An Ensembl stable ID consists of five parts: ENS(
    species)(object type)(identifier). (version). The second part is a three-letter species code. For human,
    there is no species code so IDs are in the form ENS(object type)(identifier).
    """
    if not ensembl_ids:
        return "Required argument ensembl_ids not provided!"

    if isinstance(ensembl_ids, list):
        for ensembl_id in ensembl_ids:
            if not ensembl_id.startswith("ENS"):
                return f"{ensembl_id} does not start with ENS and therefore is not an Ensembl ID!"
    elif isinstance(ensembl_ids, str):
        if not ensembl_ids.startswith("ENS"):
            return f"{ensembl_ids} does not start with ENS and therefore is not an Ensembl ID!"
    result = gget.info(
        ens_ids=ensembl_ids, ncbi=ncbi, uniprot=uniprot, pdb=pdb, verbose=True
    )
    if result is not None:
        result = result.to_string()
    return result


def gget_search(search_words, species, release=111, id_type="gene", andor="or"):
    """Fetch genes and transcripts from Ensembl using free-form search terms, for example gene names like Brca2 or
    disease names like osteoarthrititis. Results are matched based on the "gene name", "description" and "synonym"
    sections in the Ensembl database.
    """
    if not search_words:
        return "Required argument search_words not provided!"
    result = gget.search(
        searchwords=search_words,
        species=species,
        release=release,
        id_type=id_type,
        andor=andor,
        verbose=True,
    )
    if result is not None:
        result = result.to_string()
    return result


def gget_alphafold(
    sequence=None,
    multimer_for_monomer=False,
    relax=False,
    multimer_recycles=3,
    plot=True,
    show_sidechains=True,
):
    """Predicts the structure of a protein using a slightly simplified version of AlphaFold v2.3.0 (
    https://doi.org/10.1038/s41586-021-03819-2) published in the AlphaFold Colab notebook (
    https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb).
    """
    if not sequence or len(sequence) == 0:
        return "Aminoacid sequence not provided!"
    result = gget.alphafold(
        sequence=sequence,
        multimer_for_monomer=multimer_for_monomer,
        relax=relax,
        multimer_recycles=multimer_recycles,
        plot=plot,
        show_sidechains=show_sidechains,
        verbose=True,
    )
    if result is not None:
        result = result.to_string()
    return result
