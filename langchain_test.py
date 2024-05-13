from langchain_experimental.llms.ollama_functions import OllamaFunctions

model = OllamaFunctions(model="mixtral:8x22b", format="json")

model = model.bind_tools(
    tools=[
        {
            "name": "look_up_species_by_ensembl_id",
            "description": "Look up the species and database for a Ensembl ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "ensembl_id": {
                        "type": "string",
                        "description": "Stable Ensembl gene identifier"
                        "e.g ENSG00000157764",
                    },
                },
                "required": ["ensembl_id"],
            },
        }
    ],
    function_call={"name": "look_up_species_by_ensembl_id"},
)

print(model.invoke("Which species has the gene with this identifier ENSG00000157764"))
