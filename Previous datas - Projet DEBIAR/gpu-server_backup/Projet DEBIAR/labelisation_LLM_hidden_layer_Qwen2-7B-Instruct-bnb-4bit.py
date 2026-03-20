import huggingface_hub
from huggingface_hub import snapshot_download
import os
import pandas as pd
import gc
import numpy as np

from huggingface_hub import login
login("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")

import bitsandbytes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

#model_name = "unsloth/Mistral-Small-Instruct-2409-bnb-4bit"
#model_name = "unsloth/Phi-4-mini-instruct-bnb-4bit"
#model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit"
model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit"

config = AutoConfig.from_pretrained(model_name)
config.attn_implementation = "eager"
config.output_attentions = False
config.output_hidden_states = True 

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
)

model.eval()
model.to("cuda:0")

print(model.num_parameters())

import torch
import re

promptSize = 849  # longueur fixe pour découper le résultat
max_prompt_tokens = 500  # limiter le nombre de tokens des lyrics pour économiser de la mémoire

def prompt_gender(lyrics, tokenizer=None, max_prompt_tokens=None, device="cuda:0"):
    # Tronquer uniquement les lyrics si max_prompt_tokens défini
    if max_prompt_tokens is not None and tokenizer is not None:
        tokenized_lyrics = tokenizer(lyrics, return_tensors="pt").input_ids[0]
        tokenized_lyrics = tokenized_lyrics[:max_prompt_tokens].to(device)
        lyrics = tokenizer.decode(tokenized_lyrics, skip_special_tokens=True)

    return f"""You are a gender classifier that labels song lyrics based on whether the narrator appears to be male, female, or neutral. Use lyrical content, tone, and perspective to decide. Your answer must include specific words or phrases from the lyrics that influenced your decision. Return the result using this format:

LYRICS: <lyrics>  
GENDER: <male|female|neutral>  
KEYWORDS: <list of specific words or expressions from the lyrics>

EXAMPLES:

LYRICS: I wear my heart upon my sleeve, like a girl who's never been hurt before  
GENDER: female  
KEYWORDS: "girl", "wear my heart upon my sleeve"

LYRICS: Got my truck and my beer, ain't got no time for games  
GENDER: male  
KEYWORDS: "truck", "beer", "ain't got no time"

LYRICS: The sky is open, my soul is light, I drift where the wind tells me to  
GENDER: neutral  
KEYWORDS: "sky", "soul", "wind"

LYRICS: {lyrics}  
GENDER:"""

def getGenre(result):
    result = result[promptSize:]
    match = re.search(r"GENDER:\s*(male|female|neutral)", result, re.IGNORECASE)
    return match.group(1) if match else None

def getGenreLLM_with_attention_and_hidden(lyrics, model, tokenizer, device="cuda:0"):
    prompt = prompt_gender(lyrics, tokenizer=tokenizer, max_prompt_tokens=max_prompt_tokens, device=device)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.config.attn_implementation = "eager"
    model.config.output_attentions = False
    model.config.output_hidden_states = True

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, output_hidden_states=True)
        hidden_states_prompt_full = torch.stack(outputs.hidden_states).clone().detach()
        hidden_states_prompt = hidden_states_prompt_full[:,:, -1, :].cpu()

        #del prompt_without_lyrics
        del outputs, inputs, hidden_states_prompt_full
        torch.cuda.empty_cache()
        gc.collect()

    return hidden_states_prompt

# Définir le nom du modèle et le chemin de sortie
modele_str = model_name.split('/')[-1]
PATH_output_gender = f'/home/evuichard/Projet DEBIAR/labeled_lyrics_gender_{modele_str}.xlsx'

df = pd.ExcelFile(PATH_output_gender)
df = df.parse("Sheet1")

from IPython.display import clear_output
import os
import torch

# supprimer puis créer le dossier de sortie
tensor_output_dir = f"/home/evuichard/Projet DEBIAR/labeled_lyrics_tensors_{modele_str}/gender/"
if os.path.exists(tensor_output_dir):
    import shutil
    shutil.rmtree(tensor_output_dir)
os.makedirs(tensor_output_dir, exist_ok=False)

for index, row in df.iterrows():
    if index % 100 == 0:
        clear_output(wait=True)
        print(f"Processing index: {index} / {len(df)}")
        #afficher l'utilisation de la mémoire disque

    lyrics = df['english_lyrics'][index]
    genre = df['genre_LLM'][index]

    # Calculer genre + attentions + hidden states
    hidden_states_prompt = getGenreLLM_with_attention_and_hidden(lyrics, model, tokenizer)

    # Préparer le dict à sauvegarder
    data_to_save = {
        'genre': genre,
        'hidden_states_prompt': hidden_states_prompt,
    }

    # Sauvegarder un fichier .pt par ligne
    file_path = os.path.join(tensor_output_dir, f"line_{index}.pt")
    torch.save(data_to_save, file_path)

    print(f"{index} saved, genre: {genre}")

    # Nettoyage mémoire
    del hidden_states_prompt, data_to_save
    torch.cuda.empty_cache()

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

nb_tensors = 501

def load_hidden_prompt_and_gender_from_index(index):
    Path_tensor_gender = f"/home/evuichard/Projet DEBIAR/labeled_lyrics_tensors_{modele_str}/gender/" + f"/line_{index}.pt"
    data_tensor = torch.load(Path_tensor_gender)
    hidden_states = data_tensor['hidden_states_prompt'][:,0,:]
    return hidden_states, data_tensor['genre']

list_hidden_states,gender = [], []
for i in range(nb_tensors):
    hidden_states, genre = load_hidden_prompt_and_gender_from_index(i)
    list_hidden_states.append(hidden_states)
    gender.append(genre)
tensor_hidden_states = torch.stack(list_hidden_states)
del list_hidden_states
gc.collect()
torch.cuda.empty_cache()

print("Shape of tensor_hidden_states:", tensor_hidden_states.shape)

from sklearn.decomposition import PCA

def pca_on_dim_n_with_variance(tensor, dim_n=3):
    # tensor : le tenseur d'entrée de forme (nb_samples, hidden_size), ici on fera une PCA sur la dimension hidden_size
    # dim_n : le nombre de dimensions principales à conserver
    # on obtient un tenseur de forme (nb_samples, dim_n) en sortie ainsi qu'un tableau des variances expliquées
    pca = PCA(n_components=dim_n)
    reduced_tensor = pca.fit_transform(tensor.cpu().numpy())
    variance_ratio = pca.explained_variance_ratio_
    return torch.tensor(reduced_tensor, dtype=tensor.dtype, device=tensor.device), variance_ratio

variance_ratios = []
list_pca_tensors = []
for i in range(tensor_hidden_states.shape[1]):
    hidden_states = tensor_hidden_states[:,i,:]
    reduced_tensor, variance_ratio = pca_on_dim_n_with_variance(hidden_states, dim_n=3)
    list_pca_tensors.append(reduced_tensor)
    variance_ratios.append(variance_ratio)

tensor_pca = torch.stack(list_pca_tensors, dim=1)
del list_pca_tensors
print("Shape of tensor_pca:", tensor_pca.shape)

import plotly.graph_objects as go
# tensor_pca : shape [57, 501, 3]
# list_gender : 501 labels ("male", "female", "neutral")
# variances : liste de np.array shape (3,), une par couche

# Préparation des données en DataFrame
data = []
for layer in range(tensor_pca.shape[1]):
    for i in range(tensor_pca.shape[0]):
        data.append({
            "PC1": tensor_pca[i, layer, 0].item(),
            "PC2": tensor_pca[i, layer, 1].item(),
            "PC3": tensor_pca[i, layer, 2].item(),
            "Layer": layer,
            "Gender": gender[i]
        })

df = pd.DataFrame(data)

# Couleurs selon genre
color_map = {"male": "blue", "female": "red", "neutral": "green"}

# Fonction pour générer un titre d’axes avec variances
def axis_titles(layer):
    v = variance_ratios[layer] * 100  # si c’est en proportions, *100 pour %
    return dict(
        xaxis_title=f"PC1 ({v[0]:.1f}%)",
        yaxis_title=f"PC2 ({v[1]:.1f}%)",
        zaxis_title=f"PC3 ({v[2]:.1f}%)"
    )



# Figure initiale (Layer = 0)
traces = []
for g, c in color_map.items():
    mask = (df["Layer"] == 0) & (df["Gender"] == g)
    traces.append(go.Scatter3d(
        x=df.loc[mask, "PC1"],
        y=df.loc[mask, "PC2"],
        z=df.loc[mask, "PC3"],
        mode="markers",
        marker=dict(size=3, color=c),
        name=g  # légende
    ))

fig = go.Figure(data=traces)

# Frames = un layer = un nuage de points + axes mis à jour
frames = []
for layer in range(tensor_pca.shape[1]):
    traces = []
    for g, c in color_map.items():
        mask = (df["Layer"] == layer) & (df["Gender"] == g)
        traces.append(go.Scatter3d(
            x=df.loc[mask, "PC1"],
            y=df.loc[mask, "PC2"],
            z=df.loc[mask, "PC3"],
            mode="markers",
            marker=dict(size=3, color=c),
            name=g,
            showlegend=(layer == 0)  # légende affichée qu’une seule fois
        ))
    frames.append(go.Frame(data=traces, layout=dict(scene=axis_titles(layer)), name=str(layer)))

fig.frames = frames

# Ajout du slider
steps = []
for layer in range(len(frames)):
    step_dict = dict(
        method="animate",
        args=[[str(layer)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(layer)
    )
    steps.append(step_dict)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Layer: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    scene=axis_titles(0),  # initial avec layer 0
    sliders=sliders,
    width=900,   # largeur en pixels
    height=700   # hauteur en pixels
)

fig.write_html(f"/home/evuichard/Projet DEBIAR/labeled_lyrics_tensors_{modele_str}/3d_scatter_slider_gender_prompt_evolution_of_tokens_embeddings_over_layers_{modele_str}.html")
fig.show()


# Création de la figure (layer = 0)
traces = []
for g, c in color_map.items():
    mask = (df["Layer"] == 0) & (df["Gender"] == g)
    traces.append(go.Scatter3d(
        x=df.loc[mask, "PC1"],
        y=df.loc[mask, "PC2"],
        z=df.loc[mask, "PC3"],
        mode="markers",
        marker=dict(size=3, color=c),
        name=g  # affichage dans la légende
    ))

fig = go.Figure(data=traces)

# Ajout des frames (une frame = un layer/token)
frames = []
for layer in range(1, tensor_pca.shape[1]):
    traces = []
    for g, c in color_map.items():
        mask = (df["Layer"] == layer) & (df["Gender"] == g)
        traces.append(go.Scatter3d(
            x=df.loc[mask, "PC1"],
            y=df.loc[mask, "PC2"],
            z=df.loc[mask, "PC3"],
            mode="markers",
            marker=dict(size=3, color=c),
            name=g,
            showlegend=True
        ))
    frames.append(go.Frame(
        data=traces,
        layout=dict(scene=axis_titles(layer)),
        name=str(layer)
    ))

fig.frames = frames


# Boutons de contrôle
fig.update_layout(
    scene=axis_titles(0),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(label="▶ Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
        ]
    )],
    width=900,   # largeur en pixels
    height=700   # hauteur en pixels
)
fig.write_html(f"/home/evuichard/Projet DEBIAR/labeled_lyrics_tensors_{modele_str}/3d_scatter_timelapse_gender_prompt_evolution_of_tokens_embeddings_over_layers_{modele_str}.html")
fig.show()

if os.path.exists(tensor_output_dir):
    import shutil
    shutil.rmtree(tensor_output_dir)

