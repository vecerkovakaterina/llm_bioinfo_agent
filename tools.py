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
                "out": {
                    "type": "str",
                    "description": "Path to folder to save prediction results in (str)."
                    "Default: './[date_time]_gget_alphafold_prediction'",
                },
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
            "required": ["query", "reference"],
        },
    },
    {
        "name": "gget_elm",
        "description": "Locally predicts Eukaryotic Linear Motifs from an amino acid sequence or UniProt Acc using "
        "data from the ELM database (http://elm.eu.org/).",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "str",
                    "description": "Amino acid sequence or Uniprot Acc (str). If Uniprot Acc, set 'uniprot==True'.",
                },
                "uniprot": {
                    "type": "bool",
                    "description": "Set to True if the input is a Uniprot Acc instead of an amino acid sequence. "
                    "Default: False.",
                },
                "sensitivity": {
                    "type": "str",
                    "description": "Sensitivity of DIAMOND alignment. One of the following: fast, mid-sensitive, "
                    "sensitive, more-sensitive, very-sensitive, or ultra-sensitive. Default: "
                    "'very-sensitive'",
                },
                "threads": {
                    "type": "int",
                    "description": "Number of threads used in DIAMOND alignment. Default: 1",
                },
                "diamond_binary": {
                    "type": "str",
                    "description": "Path to DIAMOND binary. Default: None -> Uses DIAMOND binary installed with gget.",
                },
                "expand": {
                    "type": "bool",
                    "description": "Expand the information returned in the regex data frame to include the protein "
                    "names, organisms and references that the motif was orignally validated on. "
                    "Default: False.",
                },
                "out": {
                    "type": "str",
                    "description": "Path to folder to save results in. Default: Standard out, temporary files are "
                    "deleted.",
                },
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "gget_enrichr",
        "description": "Perform an enrichment analysis on a list of genes using Enrichr.",
        "parameters": {
            "type": "object",
            "properties": {
                "genes": {
                    "type": "list",
                    "description": "List of Entrez gene symbols to perform enrichment analysis on, passed as a list "
                    "of strings, e.g. ['PHF14', 'RBM3', 'MSL1', 'PHF21A']."
                    "Set 'ensembl = True' to input a list of Ensembl gene IDs, e.g. ["
                    "'ENSG00000106443', 'ENSG00000102317', 'ENSG00000188895'].",
                },
                "database": {
                    "type": "str",
                    "description": "Database to use as reference for the enrichment analysis."
                    "Supported shortcuts (and their default database):"
                    "'pathway' (KEGG_2021_Human)"
                    "'transcription' (ChEA_2016)"
                    "'ontology' (GO_Biological_Process_2021)"
                    "'diseases_drugs' (GWAS_Catalog_2019)"
                    "'celltypes' (PanglaoDB_Augmented_2021)"
                    "'kinase_interactions' (KEA_2015)"
                    "or any database listed under Gene-set Library at: "
                    "https://maayanlab.cloud/Enrichr/#libraries",
                },
                "background_list": {
                    "type": "list",
                    "description": "List of gene names/Ensembl IDs to be used as background genes. (Default: None)",
                },
                "background": {
                    "type": "bool",
                    "description": "If True, use set of > 20,000 default background genes listed here: "
                    "https://github.com/pachterlab/gget/blob/main/gget/constants/enrichr_bkg_genes.txt"
                    ". (Default: False)",
                },
                "ensembl": {
                    "type": "bool",
                    "description": "Define as 'True' if 'genes' is a list of Ensembl gene IDs. (Default: False)",
                },
                "ensembl_bkg": {
                    "type": "bool",
                    "description": "Define as 'True' if 'background_list' is a list of Ensembl gene IDs. (Default: "
                    "False)",
                },
                "plot": {
                    "type": "bool",
                    "description": "True/False whether to provide a graphical overview of the first 15 results. ("
                    "Default: False)",
                },
                "figsize": {
                    "type": "tuple",
                    "description": "(width, height) of plot in inches. (Default: (10,10))",
                },
                "ax": {
                    "type": "object",
                    "description": "Pass a matplotlib axes object for further customization of the plot. (Default: "
                    "None)",
                },
                "kegg_out": {
                    "type": "str",
                    "description": "Path to file to save the highlighted KEGG pathway image, "
                    "e.g. path/to/folder/kegg_pathway.png. (Default: None)",
                },
                "kegg_rank": {
                    "type": "int",
                    "description": "Candidate pathway rank to be plotted in KEGG pathway image. (Default: 1)",
                },
            },
            "required": ["genes", "database"],
        },
    },
    {
        "name": "gget_muscle",
        "description": "Align multiple nucleotide or amino acid sequences against each other (using the Muscle v5 "
        "algorithm). Returns alignment results in ClustalW formatted standard out or an 'aligned "
        "FASTA' (.afa) file.",
        "parameters": {
            "type": "object",
            "properties": {
                "fasta": {
                    "type": "list",
                    "description": "List of sequences or path to fasta file containing the sequences to be aligned.",
                },
                "super5": {
                    "type": "bool",
                    "description": "True/False (default: False). If True, align input using Super5 algorithm instead "
                    "of PPP algorithm to decrease time and memory. Use for large inputs (a few hundred"
                    " sequences).",
                },
                "out": {
                    "type": "str",
                    "description": "Path to save an 'aligned FASTA' (.afa) file with the results, "
                    "e.g. 'path/to/directory/results.afa'. Default: 'None' -> Results will be printed "
                    "in Clustal format.",
                },
            },
            "required": ["fasta"],
        },
    },
    {
        "name": "gget_mutate",
        "description": "Takes in nucleotide sequences and mutations (in standard mutation annotation - see below) and "
        "returns mutated versions of the input sequences according to the provided mutations.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequences": {
                    "type": "str|list",
                    "description": "(str) Path to the fasta file containing the sequences to be mutated, e.g., "
                    "'seqs.fa'. Sequence identifiers following the '>' character must correspond to "
                    "the identifiers in the seq_ID column of 'mutations'. NOTE: Only string until "
                    "first space or dot will be used as sequence identifier - Version numbers of "
                    "Ensembl IDs will be ignored. Example: >seq1 (or ENSG00000106443) ACTGCGATAGACT "
                    ">seq2 AGATCGCTAG Alternatively: Input sequence(s) as a string or list, "
                    "e.g. 'AGCTAGCT' or ['ACTGCTAGCT', 'AGCTAGCT'].",
                },
                "mutations": {
                    "type": "str|list",
                    "description": "Path to csv or tsv file (str) (e.g., 'mutations.csv') or data frame (DataFrame "
                    "object)"
                    "containing information about the mutations in the following format:"
                    "| mutation         | mut_ID | seq_ID |"
                    "| c.2C>T           | mut1   | seq1   | -> Apply mutation 1 to sequence 1"
                    "| c.9_13inv        | mut2   | seq2   | -> Apply mutation 2 to sequence 2"
                    "| c.9_13inv        | mut2   | seq3   | -> Apply mutation 2 to sequence 3"
                    "| c.9_13delinsAAT  | mut3   | seq3   | -> Apply mutation 3 to sequence 3"
                    "| ...              | ...    | ...    |"
                    "'mutation' = Column containing the mutations to be performed written in standard mutation "
                    "annotation (see below)"
                    "'mut_ID' = Column containing an identifier for each mutation"
                    "'seq_ID' = Column containing the identifiers of the sequences to be mutated (must correspond to "
                    "the string following"
                    "the > character in the 'sequences' fasta file; do NOT include spaces or dots)"
                    "Alternatively: Input mutation(s) as a string or list, e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C']."
                    "If a list is provided, the number of mutations must equal the number of input sequences.",
                },
                "k": {
                    "type": "int",
                    "description": "(int) Length of sequences flanking the mutation. Default: 30. If k > total length "
                    "of the sequence, the entire sequence will be kept.",
                },
                "mut_column": {
                    "type": "str",
                    "description": "Name of the column containing the mutations to be performed in mutations. "
                    "Default: 'mutation'.",
                },
                "mut_id_column": {
                    "type": "str",
                    "description": "Name of the column containing the IDs of each mutation in mutations. Default: "
                    "'mut_ID'.",
                },
                "seq_id_column": {
                    "type": "str",
                    "description": "(str) Name of the column containing the IDs of the sequences to be mutated in "
                    "'mutations'. Default: 'seq_ID'.",
                },
                "out": {
                    "type": "str",
                    "description": "Path to output FASTA file containing the mutated sequences, e.g., "
                    "'path/to/output_fasta.fa'. Default: None -> returns a list of the mutated "
                    "sequences to standard out. The identifiers (following the '>') of the mutated "
                    "sequences in the output FASTA will be '>[seq_ID]_[mut_ID]'.",
                },
            },
            "required": ["sequences", "mutations"],
        },
    },
    {
        "name": "gget_pdb",
        "description": "Query RCSB PDB for the protein structutre/metadata of a given PDB ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_id": {
                    "type": "str",
                    "description": "PDB ID to be queried (str), e.g. '7S7U'.",
                },
                "resource": {
                    "type": "str",
                    "description": "Defines type of information to be returned. 'pdb': Returns the protein structure "
                    "in PDB format (default). 'entry': Information about PDB structures at the top "
                    "level of PDB structure hierarchical data organization. 'pubmed': Get PubMed "
                    "annotations (data integrated from PubMed) for a given entry's primary citation. "
                    "'assembly': Information about PDB structures at the quaternary structure level. "
                    "'branched_entity': Get branched entity description (define entity ID as "
                    "'identifier'). 'nonpolymer_entity': Get non-polymer entity data (define entity ID "
                    "as 'identifier'). 'polymer_entity': Get polymer entity data (define entity ID as "
                    "'identifier'). 'uniprot': Get UniProt annotations for a given macromolecular "
                    "entity (define entity ID as 'identifier'). 'branched_entity_instance': Get "
                    "branched entity instance description (define chain ID as 'identifier'). "
                    "'polymer_entity_instance': Get polymer entity instance (a.k.a chain) data (define "
                    "chain ID as 'identifier'). 'nonpolymer_entity_instance': Get non-polymer entity "
                    "instance description (define chain ID as 'identifier').",
                },
                "identifier": {
                    "type": "str",
                    "description": "Can be used to define assembly, entity or chain ID if applicable (default: None)."
                    "Assembly/entity IDs are numbers (e.g. 1), and chain IDs are letters (e.g. 'A').",
                },
                "save": {
                    "type": "bool",
                    "description": "True/False wether to save JSON/PDB with query results in the current working "
                    "directory (default: False).",
                },
            },
            "required": ["pdb_id"],
        },
    },
    {
        "name": "gget_ref",
        "description": "Fetch FTPs for reference genomes and annotations by species from Ensembl.",
        "parameters": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "str",
                    "description": "Defines the species for which the reference should be fetched in the format "
                    "'<genus>_<species>', e.g. species = 'homo_sapiens'. Supported shortcuts: 'human', "
                    "'mouse', 'human_grch37' (accesses the GRCh37 genome assembly)",
                },
                "which": {
                    "type": "str",
                    "description": "Defines which results to return."
                    "Default: 'all' -> Returns all available results."
                    "Possible entries are one or a combination (as a list of strings) of the following:"
                    "'gtf' - Returns the annotation (GTF)."
                    "'cdna' - Returns the trancriptome (cDNA)."
                    "'dna' - Returns the genome (DNA)."
                    "'cds - Returns the coding sequences corresponding to Ensembl genes. (Does not contain UTR or "
                    "intronic sequence.)"
                    "'cdrna' - Returns transcript sequences corresponding to non-coding RNA genes (ncRNA)."
                    "'pep' - Returns the protein translations of Ensembl genes.",
                },
                "release": {
                    "type": "int",
                    "description": "Defines the Ensembl release number from which the files are fetched, e.g. release "
                    "= 104. Default: None -> latest Ensembl release is used",
                },
                "ftp": {
                    "type": "bool",
                    "description": "Return only the requested FTP links in a list (default: False)",
                },
                "save": {
                    "type": "bool",
                    "description": "Save the results in the local directory",
                },
                "list_species": {
                    "type": "bool",
                    "description": "If True and `species=None`, returns a list of all available VERTEBRATE species "
                    "from the Ensembl database (default: False). (Can be combined with the `release` "
                    "argument to get the available species from a specific Ensembl release.)",
                },
                "list_iv_species": {
                    "type": "bool",
                    "description": "If True and `species=None`, returns a list of all available INVERTEBRATE species "
                    "from the Ensembl database (default: False). (Can be combined with the `release` "
                    "argument to get the available species from a specific Ensembl release.)",
                },
            },
            "required": ["species"],
        },
    },
    {
        "name": "gget_seq",
        "description": "Fetch nucleotide or amino acid sequence (FASTA) of a gene (and all its isoforms) or "
        "transcript by Ensembl, WormBase or FlyBase ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "ens_ids": {
                    "type": "str|list",
                    "description": "One or more Ensembl IDs (passed as string or list of strings). Also supports "
                    "WormBase and FlyBase IDs.",
                },
                "translate": {
                    "type": "bool",
                    "description": "True/False (default: False -> returns nucleotide sequences)."
                    "Defines whether nucleotide or amino acid sequences are returned."
                    "Nucleotide sequences are fetched from the Ensembl REST API server."
                    "Amino acid sequences are fetched from the UniProt REST API server.",
                },
                "isoforms": {
                    "type": "bool",
                    "description": "If True, returns the sequences of all known transcripts (default: False).",
                },
                "save": {
                    "type": "bool",
                    "description": "If True, saves output FASTA to current directory (default: False).",
                },
            },
            "required": ["ens_ids"],
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
    out=None,
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
        out=out,
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
    if not searchterm:
        return "Required argument searchterm not provided!"
    result = gget.cosmic(
        searchterm=searchterm, entity=entity, limit=limit, out=out, verbose=True
    )
    if result is not None:
        result = result.to_markdown()
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
    if not query or not reference:
        return "Required arguments not provided!"
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


def gget_elm(
    sequence,
    uniprot=False,
    sensitivity="very-sensitive",
    threads=1,
    diamond_binary=None,
    expand=False,
    out=None,
):
    """Locally predict Eukaryotic Linear Motifs from an amino acid sequence or UniProt Acc using data from the ELM
    database."""
    if not sequence:
        return "Required argument sequence not provided!"
    result = gget.elm(
        sequence=sequence,
        uniprot=uniprot,
        sensitivity=sensitivity,
        threads=threads,
        diamond_binary=diamond_binary,
        expand=expand,
        out=out,
        verbose=True,
    )
    if result is not None:
        result = result.to_markdown()
    return result


def gget_enrichr(
    genes,
    database,
    background_list=None,
    background=False,
    ensembl=False,
    ensembl_bkg=False,
    plot=False,
    figsize=(10, 10),
    ax=None,
    kegg_out=None,
    kegg_rank=1,
):
    """Perform an enrichment analysis on a list of genes using Enrichr."""
    if not genes or not database:
        return "Required arguments not provided!"
    result = gget.enrichr(
        genes=genes,
        database=database,
        background_list=background_list,
        background=background,
        ensembl=ensembl,
        ensembl_bkg=ensembl_bkg,
        plot=plot,
        figsize=figsize,
        ax=ax,
        kegg_out=kegg_out,
        kegg_rank=kegg_rank,
        verbose=True,
    )
    if result is not None:
        result = result.to_makrdown()
    return result


def gget_muscle(fasta, super5=False, out=None):
    """Align multiple nucleotide ir amino acid sequences to each other using Muscle5. Return ClustalW formatted
    stardard out or aligned FASTA (.afa)"""
    if not fasta:
        return "Required argument fasta not provided!"
    result = gget.muscle(fasta=fasta, super5=super5, out=out, verbose=True)
    return result


def gget_mutate(
    sequences,
    mutations,
    k=30,
    mut_column="mutation",
    mut_id_column="mut_ID",
    seq_id_column="seq_ID",
    out=None,
):
    """Takes in nucleotide sequences and mutations (in standard mutation annotation - see below)
    and returns mutated versions of the input sequences according to the provided mutations.
    """
    if not sequences or not mutations:
        return "Required arguments not provided!"
    result = gget.mutate(
        seuqnces=sequences,
        mutations=mutations,
        k=k,
        mut_column=mut_column,
        mut_id_column=mut_id_column,
        seq_id_column=seq_id_column,
        out=out,
        verbose=True,
    )
    return result


def gget_pdb(pdb_id, resource="pdb", identifier=None, save=False):
    """Query RCSB Protein Data Bank (PDB) for the protein structure/metadata of a given PDB ID."""
    if not pdb_id:
        return "Required argument pdb_id not provided!"
    result = gget.pdb(
        pdb_id=pdb_id, resource=resource, identifier=identifier, save=save
    )
    return result


def gget_ref(
    species,
    which="all",
    release=None,
    ftp=False,
    save=False,
    list_species=False,
    list_iv_species=False,
):
    """Fetch FTPs for reference genomes and annotations by species from Ensembl."""
    if not species:
        return "Required argument species not provided!"
    result = gget.ref(
        species=species,
        which=which,
        release=release,
        ftp=ftp,
        save=save,
        list_species=list_species,
        list_iv_species=list_iv_species,
        verbose=True,
    )
    return result


def gget_seq(ens_ids, translate=False, isoforms=False, save=False):
    """Fetch nucleotide or amino acid sequence (FASTA) of a gene
    (and all its isoforms) or transcript by Ensembl, WormBase or FlyBase ID."""
    if not ens_ids:
        return "Required argument ens_ids not provided!"
    result = gget.seq(
        ens_ids=ens_ids,
        translate=translate,
        isoforms=isoforms,
        save=save,
        verbose=True,
    )
    return result
