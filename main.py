# Comments rah mkhelten chwya darija chwya francais chwya eng
import os
import json
import numpy as np
import cv2
from glob import glob
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

# ============================
# Utilities: read & preprocess
# ============================

def load_and_resize(image_path, size=(256,256)):
    """
    hadi fonction katsir l'image mn disk, katdir resize w katb9a RGB
    // Cette fonction charge l'image, la redimensionne et la convertit en RGB.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible d'ouvrir {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def compute_histogram_rgb(img):
    """
    kat7seb histogramme complet 256 bins per channel
    // Retourne array shape (3,256)
    """
    chans = cv2.split(img)
    hist = [cv2.calcHist([c], [0], None, [256], [0,256]).flatten() for c in chans]
    hist = np.array(hist)  # shape (3,256)
    # normaliser par somme totale
    s = hist.sum()
    if s > 0:
        hist = hist / s
    return hist

def histobine_from_histogram(hist, bins_per_channel=16):
    """
    reduc hist (256 -> bins_per_channel) by grouping levels
    // hist: shape (3,256) returns vector (3*bins_per_channel,)
    """
    group = 256 // bins_per_channel
    small = []
    for c in range(3):
        arr = hist[c]
        new = [arr[i*group:(i+1)*group].sum() for i in range(bins_per_channel)]
        small.extend(new)
    small = np.array(small, dtype=float)
    s = small.sum()
    if s > 0:
        small = small / s
    return small  # shape (3*bins_per_channel,)

def compute_histobine(img, bins_per_channel=16):
    """
    direct pipeline: load img matrix -> full hist -> histobine (normalized)
    """
    hist = compute_histogram_rgb(img)
    return histobine_from_histogram(hist, bins_per_channel=bins_per_channel)

# ============================
# Distances / Similarities
# ============================

def hist_intersection(h1, h2):
    """
    Swain & Ballard histogram intersection similarity
    // h1, h2 should be normalized vectors; return score in [0,1], bigger = more similar
    """
    return np.minimum(h1, h2).sum()

def euclidean_distance(h1, h2):
    return np.linalg.norm(h1 - h2)

def chi2_distance(h1, h2, eps=1e-10):
    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    return 0.5 * np.sum(num / den)

def correlation_score(h1, h2):
    # if variance zero, return minimal
    if np.std(h1) == 0 or np.std(h2) == 0:
        return 0.0
    return np.corrcoef(h1, h2)[0,1]

# ============================
# Indexation (Phase hors-ligne)
# ============================

def build_index(dataset_dir, out_json="descripteurs.json", bins_per_channel=16, size=(256,256), force_rebuild=False):
    """
    Hayda script li ydir indexation:
    - parcourir dataset_dir
    - pour chaque image calculer histobine
    - sauvegarder dict {filename: vector}
    """
    out_path = Path(out_json)
    if out_path.exists() and not force_rebuild:
        print(f"{out_json} exists — use force_rebuild=True to overwrite")
        return

    img_paths = sorted(glob(os.path.join(dataset_dir, "**", "*.*"), recursive=True))  # kdkhel lwest dataset ofwest subfolders
    descriptors = {}
    for p in img_paths:
        try:
            img = load_and_resize(p, size=size)
            vec = compute_histobine(img, bins_per_channel=bins_per_channel)
            descriptors[os.path.relpath(p, dataset_dir)] = vec.tolist()
        except Exception as e:
            print("Erreur pour", p, e)

    with open(out_json, "w") as f:
        json.dump(descriptors, f)
    print("Index built:", len(descriptors), "entries")
    return descriptors

def load_index(json_path="descripteurs.json"):
    with open(json_path, "r") as f:
        d = json.load(f)
    # convert values to numpy arrays
    return {k: np.array(v, dtype=float) for k,v in d.items()}

# ============================
# Search function (phase en-ligne)
# ============================

def search(query_img, index, bins_per_channel=16, measure="intersection", top_k=10):
    """
    query_img: numpy image matrix (RGB)
    index: dict {filename: np.array(vec)}
    returns sorted results [(score, filename), ...]
    """
    q = compute_histobine(query_img, bins_per_channel=bins_per_channel)
    results = []
    for fname, vec in index.items():
        if measure == "intersection":
            score = hist_intersection(q, vec)
            # bigger = more similar -> we'll sort descending
            results.append((score, fname))
        elif measure == "euclidean":
            score = euclidean_distance(q, vec)
            results.append((score, fname))
        elif measure == "chi2":
            score = chi2_distance(q, vec)
            results.append((score, fname))
        elif measure == "correlation":
            score = correlation_score(q, vec)
            results.append((score, fname))
        else:
            raise ValueError("Unknown measure")
    # sort: for intersection & correlation desc, for distances asc
    if measure in ["intersection", "correlation"]:
        results.sort(key=lambda x: x[0], reverse=True)
    else:
        results.sort(key=lambda x: x[0])  # smaller is better
    return results[:top_k]

# ============================
# Streamlit UI
# ============================

def show_image_grid(image_paths, cols=5, title="Results"):
    st.write("###", title)
    rows = (len(image_paths) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colsui = st.columns(cols)
        for c in range(cols):
            if idx >= len(image_paths): break
            with colsui[c]:
                st.image(image_paths[idx], use_column_width='always')
                st.caption(Path(image_paths[idx]).name)
            idx += 1

def main():
    st.title("Mini-Projet- CBIR (Indexation et Recherche) - Lamraoui Gi4")

    st.sidebar.header("Index / Params")
    dataset_dir = st.sidebar.text_input("Dataset dir", value="dataset")
    json_path = st.sidebar.text_input("Descriptors JSON", value="descripteurs.json")
    bins = st.sidebar.slider("bins per channel", 8, 32, 16, step=8)
    measure = st.sidebar.selectbox("Distance / Similarity", ["intersection","euclidean","chi2","correlation"])
    top_k = st.sidebar.slider("Top K", 1, 20, 9)

    if st.sidebar.button("Build index (scan dataset)"):
        with st.spinner("Building index..."):
            build_index(dataset_dir, out_json=json_path, bins_per_channel=bins, force_rebuild=True)
        st.success("Index built")

    if Path(json_path).exists():
        index = load_index(json_path)
        st.sidebar.write("Index entries:", len(index))
    else:
        index = {}
        st.sidebar.warning("No index file found")

    uploaded = st.file_uploader("Upload query image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded is not None:
        # read as opencv image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Query image", use_column_width=True)

        if st.button("Search"):
            if not index:
                st.error("Index empty — build it first")
            else:
                results = search(img, index, bins_per_channel=bins, measure=measure, top_k=top_k)
                # prepare list of file paths to show (we assume dataset/<filename>)
                res_paths = [os.path.join(dataset_dir, fname) for (_, fname) in results]
                show_image_grid(res_paths, cols=3, title="Top results")

if __name__ == "__main__":
    main()
