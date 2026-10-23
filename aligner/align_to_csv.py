import os
from tqdm import tqdm
import re

def normalize_text(text):
    """Normalize text by removing special characters and normalizing spaces"""
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def create_alignment_pairs_csv():
    # Chemins finaux
    src_file = 'data/lancelot/source.txt'
    tgt_file = 'data/lancelot/target.txt'
    infer_output_dir = 'infer_output'
    output_csv = 'alignment_pairs.csv'

    # Fichier spécifique à utiliser
    specific_align_file = 'XX2XX.align.6'

    # Lire les fichiers source et cible
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [normalize_text(line.strip()) for line in f.readlines()]

    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [normalize_text(line.strip()) for line in f.readlines()]

    # Vérifier que les fichiers ont le même nombre de lignes
    if len(src_sentences) != len(tgt_sentences):
        raise ValueError("Les fichiers source et cible doivent avoir le même nombre de lignes")

    # Vérifier que le fichier spécifique existe
    specific_align_path = os.path.join(infer_output_dir, specific_align_file)
    if not os.path.exists(specific_align_path):
        raise FileNotFoundError("Le fichier spécifique %s n'existe pas dans %s" % (specific_align_file, infer_output_dir))

    # Lire le fichier d'alignement spécifique
    with open(specific_align_path, 'r', encoding='utf-8') as f:
        align_lines = f.readlines()

    # Créer le fichier CSV
    with open(output_csv, 'w', encoding='utf-8') as csv_file:
        # Écrire l'en-tête
        csv_file.write('source_word,target_word\n')

        for i, (src_sent, tgt_sent) in tqdm(enumerate(zip(src_sentences, tgt_sentences)), total=len(src_sentences)):
            if i < len(align_lines):
                # Convertir les alignements en paires de mots
                align_pairs = align_lines[i].strip().split()
                for pair in align_pairs:
                    if '-' in pair:
                        src_idx, tgt_idx = pair.split('-')
                        try:
                            src_word = src_sent.split()[int(src_idx)]
                            tgt_word = tgt_sent.split()[int(tgt_idx)]
                            # Écrire chaque paire d'alignement sur une ligne séparée
                            csv_file.write("\"%s\",\"%s\"\n" % (src_word, tgt_word))
                        except (IndexError, ValueError):
                            continue  # Skip invalid indices

    print("Fichier CSV créé avec succès : %s" % output_csv)

# Exécuter la fonction
if __name__ == "__main__":
    create_alignment_pairs_csv()