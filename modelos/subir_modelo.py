from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="modelos/transformer_pico_placa.pt",
    path_in_repo="transformer_pico_placa.pt",
    repo_id="Huntercito/Deteccion_Pico_y_Placa",
    repo_type="model"
)
