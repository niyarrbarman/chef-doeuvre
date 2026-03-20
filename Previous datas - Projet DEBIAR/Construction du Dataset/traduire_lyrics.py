import pandas as pd
import time
import deepl

DEEPL_API_KEY = "4b2344ff-6714-4bc1-b311-71534de0652c:fx"  # Remplace par ta vraie clé API

# Initialiser le traducteur DeepL
deepl_translator = deepl.Translator(DEEPL_API_KEY)
CHARACTER_LIMIT = 500_000
character_count = 0

#INPUT_XLSX = "spotify_songs_with_regions_and_gender.xlsx"
INPUT_XLSX = "translated_lyrics.xlsx"
OUTPUT_XLSX = "translated_lyrics.xlsx"

# Charger le fichier Excel
df = pd.read_excel(INPUT_XLSX)
df["english_lyrics"] = None
#0, 145,152,358,375,404,538,683,915,990,1155 erreur source
for idx, row in df.iterrows():
    original_lang = str(row["language"]).lower()
    lyrics = row["lyrics"]

    if pd.isna(lyrics) or original_lang == "en":
        continue

    if character_count + len(lyrics) > CHARACTER_LIMIT:
        print("Limite de traduction DeepL atteinte.")
        break

    try:
        time.sleep(0.3)  # Pour éviter de surcharger l'API
        result = deepl_translator.translate_text(lyrics, target_lang="EN-US")
        translation = result.text
        character_count += len(lyrics)
        df.at[idx, "english_lyrics"] = translation
    except Exception as e:
        print(f"Erreur DeepL pour l'entrée {idx} : {e}")
        continue
    print("Ligne ", idx, " traduite")
    

# Sauvegarder le fichier Excel
df.to_excel(OUTPUT_XLSX, index=False)
print(f"Traductions terminées. Résultat enregistré dans {OUTPUT_XLSX}")
