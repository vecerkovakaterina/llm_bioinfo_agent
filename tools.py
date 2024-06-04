import json

import gget
import pandas as pd
from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION

tools = [
    {
        "name": "get_further_clarification",
        "description": "Ask user for missing information, further clarification, instructions or more details",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "str",
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
                    "type": "str",
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
                    "type": "str",
                    "description": "'gene' (default) or 'transcript' Defines whether genes or transcripts matching "
                    "the searchwords are returned.",
                },
                "andor": {
                    "type": "str",
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
                    "type": "str",
                    "description": "Amino acid sequence (str), a list of sequences",
                },
                # "out": {
                #     "type": "str",
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
    {
        "name": "gget_archs4",
        "description": "Find the most correlated genes or the tissue expression atlas of a gene of interest using "
        "data from the human and mouse RNA-seq database ARCHS4 (https://maayanlab.cloud/archs4/).",
        "parameters": {
            "type": "object",
            "properties": {
                "gene": {
                    "type": "str",
                    "description": "Short name (Entrez gene symbol) of gene of interest (str), e.g. 'STAT4'. Set "
                    "'ensembl=True' to input an Ensembl gene ID, e.g. ENSG00000138378.",
                },
                "ensembl": {
                    "type": "bool",
                    "description": "Define as 'True' if 'gene' is an Ensembl gene ID. (Default: False)",
                },
                "which": {
                    "type": "str",
                    "description": "'correlation' (default) or 'tissue'. - 'correlation' returns a gene correlation "
                    "table that contains the 100 most correlated genes to the gene of interest. The "
                    "Pearson correlation is calculated over all samples and tissues in ARCHS4. - "
                    "'tissue' returns a tissue expression atlas calculated from human or mouse samples "
                    "(as defined by 'species') in ARCHS4.",
                },
                "gene_count": {
                    "type": "int",
                    "description": "Number of correlated genes to return (default: 100). (Only for gene correlation.)",
                },
                "species": {
                    "type": "int",
                    "description": "'human' (default) or 'mouse'. (Only for tissue expression atlas.)",
                },
            },
            "required": ["gene"],
        },
    },
    {
        "name": "gget_blast",
        "description": "BLAST a nucleotide or amino acid sequence against any BLAST DB.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "str",
                    "description": "Sequence (str) or path to FASTA file.",
                },
                "program": {
                    "type": "str",
                    "description": "'blastn', 'blastp', 'blastx', 'tblastn', or 'tblastx'. Default: 'blastn' for "
                    "nucleotide sequences; 'blastp' for amino acid sequences.",
                },
                "database": {
                    "type": "str",
                    "description": "'nt', 'nr', 'refseq_rna', 'refseq_protein', 'swissprot', 'pdbaa', or 'pdbnt'. "
                    "Default: 'nt' for nucleotide sequences; 'nr' for amino acid sequences.",
                },
                "limit": {
                    "type": "int",
                    "description": "Limits number of hits to return. Default 50.",
                },
                "expect": {
                    "type": "float",
                    "description": "float or None. An expect value cutoff. Default 10.0.",
                },
                "low_comp_filt": {
                    "type": "bool",
                    "description": "True/False whether to apply low complexity filter. Default False.",
                },
                "megablast": {
                    "type": "bool",
                    "description": "True/False whether to use the MegaBLAST algorithm (blastn only). Default True.",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "gget_blat",
        "description": "BLAT a nucleotide or amino acid sequence against any BLAT UCSC assembly.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "str",
                    "description": "Sequence (str) or path to fasta file containing one sequence.",
                },
                "seqtype": {
                    "type": "str",
                    "description": "'DNA', 'protein', 'translated%20RNA', or 'translated%20DNA'. Default: 'DNA' for "
                    "nucleotide sequences; 'protein' for amino acid sequences.",
                },
                "assembly": {
                    "type": "str",
                    "description": "'human' (hg38) (default), 'mouse' (mm39), 'zebrafinch' (taeGut2), or any of the "
                    "species assemblies available at https://genome.ucsc.edu/cgi-bin/hgBlat (use short "
                    "assembly name as listed after the' / ').",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "gget_cellxgene",
        "description": "Query data from CZ CELLxGENE Discover (https://cellxgene.cziscience.com/) using the CZ "
        "CELLxGENE Discover Census (https://github.com/chanzuckerberg/cellxgene-census). NOTE: "
        "Querying large datasets requires a large amount of RAM. Use the cell metadata attributes to "
        "define the (sub)dataset of interest. The CZ CELLxGENE Discover Census recommends >16 GB of "
        "memory and a >5 Mbps internet connection. Returns AnnData object (when meta_only=False) or dataframe (when "
        "meta_only=True).",
        "parameters": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "str",
                    "description": "Choice of 'homo_sapiens' or 'mus_musculus'. Default: 'homo_sapiens'.",
                },
                "gene": {
                    "type": "str|list",
                    "description": "Str or list of gene name(s) or Ensembl ID(s), e.g. ['ACE2', 'SLC5A1'] or ["
                    "'ENSG00000130234', 'ENSG00000100170']. Default: None. NOTE: Set ensembl=True when "
                    "providing Ensembl ID(s) instead of gene name(s).",
                },
                "ensembl": {
                    "type": "bool",
                    "description": "True/False (default: False). Set to True when genes are provided as Ensembl IDs.",
                },
                "column_names": {
                    "type": "list",
                    "description": "List of metadata columns to return (stored in AnnData.obs when meta_only=False). "
                    "Default: ['dataset_id', 'assay', 'suspension_type', 'sex', 'tissue_general', "
                    "'tissue', 'cell_type']",
                },
                "meta_only": {
                    "type": "bool",
                    "description": "True/False (default: False). If True, returns only metadata dataframe ("
                    "corresponds to AnnData.obs).",
                },
                "census_version": {
                    "type": "str",
                    "description": "Str defining version of Census, e.g. '2023-05-15' or 'latest' or 'stable'. "
                    "Default: 'stable'.",
                },
                "out": {
                    "type": "str",
                    "description": "If provided, saves the generated AnnData h5ad (or csv when meta_only=True) file "
                    "with the specified path. Default: None.",
                },
                "tissue": {
                    "type": "str|list",
                    "description": "Str or list of tissue(s), e.g. ['lung', 'blood']. Default: None.",
                },
                "cell_type": {
                    "type": "str|list",
                    "description": "Str or list of celltype(s), e.g. ['mucus secreting cell', 'neuroendocrine cell']. "
                    "Default: None.",
                },
                "development_stage": {
                    "type": "str|list",
                    "description": "Str or list of development stage(s). Default: None.",
                },
                "disease": {
                    "type": "str|list",
                    "description": "Str or list of disease(s). Default: None.",
                },
                "sex": {
                    "type": "str|list",
                    "description": "Str or list of sex(es), e.g. 'female'. Default: None.",
                },
                "is_primary_data": {
                    "type": "bool",
                    "description": "True/False (default: True). If True, returns only the canonical instance of the "
                    "cellular observation. This is commonly set to False for meta-analyses reusing "
                    "data or for secondary views of data.",
                },
                "dataset_id": {
                    "type": "str|list",
                    "description": "Str or list of CELLxGENE dataset ID(s). Default: None.",
                },
                "tissue_general_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of high-level tissue UBERON ID(s). Default: None.",
                },
                "tissue_general": {
                    "type": "str|list",
                    "description": "Str or list of high-level tissue label(s). Default: None.",
                },
                "tissue_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of tissue ontology term ID(s) as defined in the CELLxGENE dataset "
                    "schema. Default: None.",
                },
                "assay_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of assay ontology term ID(s) as defined in the CELLxGENE dataset "
                    "schema. Default: None.",
                },
                "assay": {
                    "type": "str|list",
                    "description": "Str or list of assay(s) as defined in the CELLxGENE dataset schema. Default: None.",
                },
                "cell_type_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of celltype ontology term ID(s) as defined in the CELLxGENE dataset "
                    "schema. Default: None.",
                },
                "development_stage_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of development stage ontology term ID(s) as defined in the CELLxGENE "
                    "dataset schema. Default: None.",
                },
                "disease_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of disease ontology term ID(s) as defined in the CELLxGENE dataset "
                    "schema. Default: None.",
                },
                "donor_id": {
                    "type": "str|list",
                    "description": "Str or list of donor ID(s) as defined in the CELLxGENE dataset schema. Default: "
                    "None.",
                },
                "self_reported_ethnicity_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of self reported ethnicity ontology ID(s) as defined in the CELLxGENE "
                    "dataset schema. Default: None.",
                },
                "self_reported_ethnicity": {
                    "type": "str|list",
                    "description": "Str or list of self reported ethnicity as defined in the CELLxGENE dataset "
                    "schema. Default: None.",
                },
                "sex_ontology_term_id": {
                    "type": "str|list",
                    "description": "Str or list of sex ontology ID(s) as defined in the CELLxGENE dataset schema. "
                    "Default: None.",
                },
                "suspension_type": {
                    "type": "str|list",
                    "description": "Str or list of suspension type(s) as defined in the CELLxGENE dataset schema. "
                    "Default: None.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "gget_cosmic",
        "description": "Search for genes, mutations, etc associated with cancers using the COSMIC (Catalogue Of "
        "Somatic Mutations In Cancer) database",
        "parameters": {
            "type": "object",
            "properties": {
                "searchterm": {
                    "type": "str",
                    "description": "(str) Search term, which can be a mutation, gene name (or Ensembl ID), sample, etc."
                    "Examples for the searchterm and entitity arguments:"
                    "| searchterm   | entitity    |"
                    "|--------------|-------------|"
                    "| EGFR         | mutations   | -> Find mutations in the EGFR gene that are associated with cancer"
                    "| v600e        | mutations   | -> Find genes for which a v600e mutation is associated with cancer"
                    "| COSV57014428 | mutations   | -> Find mutations associated with this COSMIC mutations ID"
                    "| EGFR         | genes       | -> Get the number of samples, coding/simple mutations, and fusions observed in COSMIC for EGFR"
                    "| prostate     | cancer      | -> Get number of tested samples and mutations for prostate cancer"
                    "| prostate     | tumour_site | -> Get number of tested samples, genes, mutations, fusions, etc. with 'prostate' as primary tissue site"
                    "| ICGC         | studies     | -> Get project code and descriptions for all studies from the ICGC (International Cancer Genome Consortium)"
                    "| EGFR         | pubmed      | -> Find PubMed publications on EGFR and cancer"
                    "| ICGC         | samples     | -> Get metadata on all samples from the ICGC (International Cancer Genome Consortium)"
                    "| COSS2907494  | samples     | -> Get metadata on this COSMIC sample ID (cancer type, tissue, # analyzed genes, # mutations, etc.)",
                },
                "entity": {
                    "type": "str",
                    "description": "Defines the type of the results to return. One of the following: 'mutations' ("
                    "default), 'genes', 'cancer', 'tumour_site', 'studies', 'pubmed', or 'samples'.",
                },
                "limit": {
                    "type": "int",
                    "description": "Number of hits to return. Default: 100",
                },
                "out": {
                    "type": "str",
                    "description": "Path to the file the results will be saved in, e.g. 'path/to/results.json'. "
                    "Default: None",
                },
            },
            "required": ["searchterm"],
        },
    },
    {
        "name": "gget_diamond",
        "description": "Align multiple protein or translated DNA sequences using DIAMOND (DIAMOND is similar to "
                       "BLAST, but this is a local computation).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "str|list",
                    "description": "Sequences (str or list) or path to FASTA file containing sequences to be aligned "
                    "against the reference.",
                },
                "reference": {
                    "type": "str|list",
                    "description": "Reference sequences (str or list) or path to FASTA file containing reference "
                    "sequences.",
                },
                "diamond_db": {
                    "type": "str",
                    "description": "Path to save DIAMOND database created from reference. Default: None -> Temporary "
                    "db file will be deleted after alignment or saved in 'out' if 'out' is provided.",
                },
                "sensitivity": {
                    "type": "str",
                    "description": "Sensitivity of DIAMOND alignment. One of the following: fast, mid-sensitive, "
                    "sensitive, more-sensitive, very-sensitive or ultra-sensitive. Default: "
                    "'very-sensitive'",
                },
                "threads": {
                    "type": "int",
                    "description": "Number of threads to use for alignment. Default: 1.",
                },
                "diamond_binary": {
                    "type": "str",
                    "description": "Path to DIAMOND binary, e.g. path/bins/Linux/diamond. Default: None -> Uses "
                    "DIAMOND binary installed with gget.",
                },
                "out": {
                    "type": "str",
                    "description": "Path to folder to save DIAMOND results in. Default: Standard out, temporary files "
                    "are deleted.",
                },
            },
            "required": ["query, reference"],
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
        result = result.to_markdown()
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
        result = result.to_markdown()
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
    if not sequence:
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
        result = result.to_markdown()
    return result


def gget_archs4(
    gene, ensembl=False, which="correlation", gene_count=100, species="human"
):
    """Find the most correlated genes or the tissue expression atlas
    of a gene of interest using data from the human and mouse RNA-seq
    database ARCHS4 (https://maayanlab.cloud/archs4/)."""
    if not gene:
        return "Required argument gene not provided!"
    result = gget.archs4(
        gene=gene,
        ensembl=ensembl,
        which=which,
        gene_count=gene_count,
        species=species,
        verbose=True,
    )
    if result is not None:
        result = result.to_markdown()
    return result


def gget_blast(
    sequence,
    program="default",
    database="default",
    limit=50,
    expect=10.0,
    low_comp_filt=False,
    megablast=True,
):
    """BLAST a nucleotide or amino acid sequence against any BLAST DB."""
    if not sequence:
        return "Required argument sequence not provided!"
    result = gget.blast(
        sequence=sequence,
        program=program,
        database=database,
        limit=limit,
        expect=expect,
        low_comp_filt=low_comp_filt,
        megablast=megablast,
        verbose=True,
    )
    if result is not None:
        result = result.to_markdown()
    return result


def gget_blat(sequence, seqtype="default", assembly="human"):
    """BLAT a nucleotide or amino acid sequence against any BLAT UCSC assembly."""
    if not sequence:
        return "Required argument sequence not provided!"
    result = gget.blat(
        sequence=sequence, seqtype=seqtype, assembly=assembly, verbose=True
    )
    if result is not None:
        result = result.to_markdown()
    return result


def gget_cellxgene(
    species="homo_sapiens",
    gene=None,
    ensembl=False,
    column_names=[
        "dataset_id",
        "assay",
        "suspension_type",
        "sex",
        "tissue_general",
        "tissue",
        "cell_type",
    ],
    meta_only=False,
    tissue=None,
    cell_type=None,
    development_stage=None,
    disease=None,
    sex=None,
    is_primary_data=True,
    dataset_id=None,
    tissue_general_ontology_term_id=None,
    tissue_general=None,
    assay_ontology_term_id=None,
    assay=None,
    cell_type_ontology_term_id=None,
    development_stage_ontology_term_id=None,
    disease_ontology_term_id=None,
    donor_id=None,
    self_reported_ethnicity_ontology_term_id=None,
    self_reported_ethnicity=None,
    sex_ontology_term_id=None,
    suspension_type=None,
    tissue_ontology_term_id=None,
    census_version="stable",
    out=None,
):
    """Query data from CZ CELLxGENE Discover (https://cellxgene.cziscience.com/) using the
    CZ CELLxGENE Discover Census (https://github.com/chanzuckerberg/cellxgene-census).

    NOTE: Querying large datasets requires a large amount of RAM. Use the cell metadata attributes
    to define the (sub)dataset of interest.
    The CZ CELLxGENE Discover Census recommends >16 GB of memory and a >5 Mbps internet connection.
    """
    result = gget.cellxgene(
        species=species,
        gene=gene,
        ensembl=ensembl,
        column_names=column_names,
        meta_only=meta_only,
        tissue=tissue,
        cell_type=cell_type,
        development_stage=development_stage,
        disease=disease,
        sex=sex,
        is_primary_data=is_primary_data,
        dataset_id=dataset_id,
        tissue_general_ontology_term_id=tissue_general_ontology_term_id,
        tissue_general=tissue_general,
        assay_ontology_term_id=assay_ontology_term_id,
        assay=assay,
        cell_type_ontology_term_id=cell_type_ontology_term_id,
        development_stage_ontology_term_id=development_stage_ontology_term_id,
        disease_ontology_term_id=disease_ontology_term_id,
        donor_id=donor_id,
        self_reported_ethnicity_ontology_term_id=self_reported_ethnicity_ontology_term_id,
        self_reported_ethnicity=self_reported_ethnicity,
        sex_ontology_term_id=sex_ontology_term_id,
        suspension_type=suspension_type,
        tissue_ontology_term_id=tissue_ontology_term_id,
        census_version=census_version,
        verbose=True,
        out=out,
    )
    if result is not None:
        if isinstance(result, pd.DataFrame):
            result = result.to_markdown()
        else:
            # TODO returns AnnData object when meta_only=False
            pass
    return result


def gget_cosmic(
    searchterm,
    entity="mutations",
    limit=100,
    out=None,
):
    """Search for genes, mutations, and other factors associated with cancer using the COSMIC (Catalogue Of Somatic
    Mutations In Cancer) database."""
    result = gget.cosmic()
    if result is not None:
        result = result.to_markdown(
            searchterm=searchterm, entity=entity, limit=limit, out=out, verbose=True
        )
    return result


def gget_diamond(
    query,
    reference,
    diamond_db=None,
    sensitivity="very-sensitive",
    threads=1,
    diamond_binary=None,
    out=None,
):
    """Align multiple protein or translated DNA sequences using DIAMOND (DIAMOND is similar to BLAST, but this is a
    local computation)."""
    result = gget.diamond(
        query=query,
        reference=reference,
        diamond_db=diamond_db,
        sensitivity=sensitivity,
        threads=threads,
        diamond_binary=diamond_binary,
        out=out,
        verbose=True,
    )
    if result is not None:
        result.to_markdown()
    return result
