import streamlit as st
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    mdl.eval()
    return mdl, tok

model, tokenizer = load_llm()

# Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    df = pd.read_csv(CSV_FILE)
    return df["processed"].dropna().astype(str).unique().tolist()

ingredient_list = load_ingredient_data()


# Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings(_ingredient_list):
    filtered = []
    vectors = []
    for ing in _ingredient_list:
        txt = ing.strip().lower()
        if not txt:
            continue
        vec = nlp(txt).vector
        if np.any(vec):# Exclude zero vectors
            filtered.append(ing)
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32), filtered

ingredient_vectors, filtered_ingredient_list = compute_embeddings(ingredient_list)


# Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  #  Uses angular distance (1 - cosine similarity)
    for i, vec in enumerate(vectors):
        index.add_item(i, vec)
    index.build(50)  #  More trees = better accuracy
    return index

annoy_index = build_annoy_index(ingredient_vectors)

#  Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def direct_search_alternatives(ingredient: str, top_n: int = 3):
    query = ingredient.strip().lower()
    if not query:
        return ["No alternatives found"] * top_n

    query_vec = nlp(query).vector
    if not np.any(query_vec):
        return ["No alternatives found"] * top_n

    sims = []
    for i, vec in enumerate(ingredient_vectors):
        sims.append((cosine_similarity(query_vec, vec), i))

    sims.sort(reverse=True, key=lambda x: x[0])

    results = []
    for sim, idx in sims:
        cand = filtered_ingredient_list[idx]
        if cand.strip().lower() == query:
            continue
        results.append(cand)
        if len(results) >= top_n:
            break

    while len(results) < top_n:
        results.append("No alternatives found")
    return results

#  Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient: str, top_n: int = 3):
    query = ingredient.strip().lower()
    if not query:
        return ["No alternatives found"] * top_n

    query_vec = nlp(query).vector
    if not np.any(query_vec):
        return ["No alternatives found"] * top_n

    ids = annoy_index.get_nns_by_vector(query_vec, top_n + 5)  # extra to skip exact match
    results = []
    for idx in ids:
        cand = filtered_ingredient_list[idx]
        if cand.strip().lower() == query:
            continue
        results.append(cand)
        if len(results) >= top_n:
            break

    while len(results) < top_n:
        results.append("No alternatives found")
    return results

#System prompt styles
SYSTEM_PROMPTS = {
    "Structured": (
        "You are ChefBot.\n"
        "Return the recipe in this exact structure (no extra sections):\n"
        "TITLE:\n"
        "INGREDIENTS (bullet list):\n"
        "INSTRUCTIONS (numbered steps):\n"
    ),
    "Concise": (
        "You are ChefBot.\n"
        "Return a short recipe:\n"
        "- Title (one line)\n"
        "- Ingredients: max 6 bullets\n"
        "- Instructions: max 5 steps\n"
        "No extra text.\n"
    ),
    "Creative": (
        "You are ChefBot.\n"
        "Create a creative recipe with a surprising but tasty twist.\n"
        "Use: TITLE, INGREDIENTS (bullets), INSTRUCTIONS (numbered).\n"
        "End with ONE line: Creative Twist: ...\n"
    ),
}

# Dietary preference + servings + nutrition estimate
DIET_OPTIONS = [
    "None",
    "Vegetarian",
    "Vegan",
    "Gluten-free",
    "Dairy-free",
    "Low-carb",
    "High-protein",
]

def estimate_nutrition_llm(recipe_text: str, servings: int):
    """
    Returns (nutrition_dict, raw_text_if_failed).

    Fixes the 'all 2.0000' bug by parsing ONLY labeled lines (not the echoed prompt).
    """
    prompt = (
        "You are a nutrition estimator.\n"
        "Estimate APPROXIMATE nutrition PER SERVING from the recipe.\n"
        "Return EXACTLY 7 lines, each in this format:\n"
        "Calories (kcal): <number>\n"
        "Carbs (g): <number>\n"
        "Protein (g): <number>\n"
        "Fat (g): <number>\n"
        "Fiber (g): <number>\n"
        "Sugar (g): <number>\n"
        "Sodium (mg): <number>\n"
        "No extra text.\n\n"
        f"Servings: {servings}\n"
        "Recipe:\n"
        f"{recipe_text}\n\n"
        "Nutrition per serving:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=160,
            do_sample=True,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    marker = "Nutrition per serving:"
    if marker in decoded:
        decoded = decoded.split(marker, 1)[-1]

    def parse_line(label, text):
        m = re.search(
            rf"^{re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        return float(m.group(1)) if m else None

    nutrition = {
        "calories_kcal": parse_line("Calories (kcal)", decoded),
        "carbs_g": parse_line("Carbs (g)", decoded),
        "protein_g": parse_line("Protein (g)", decoded),
        "fat_g": parse_line("Fat (g)", decoded),
        "fiber_g": parse_line("Fiber (g)", decoded),
        "sugar_g": parse_line("Sugar (g)", decoded),
        "sodium_mg": parse_line("Sodium (mg)", decoded),
    }

    if all(v is not None for v in nutrition.values()):
        return nutrition, None

    return None, decoded

#  Generate Recipe
def generate_recipe(
    ingredients: str,
    cuisine: str,
    system_prompt: str,
    dietary_pref: str,
    servings: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    decoding_strategy: str,
    max_new_tokens: int,
):
    ing_list = [x.strip() for x in ingredients.split(",") if x.strip()]
    ing_text = ", ".join(ing_list) if ing_list else ingredients.strip()

    diet_line = "No dietary restrictions." if dietary_pref == "None" else f"Dietary preference: {dietary_pref}."

    prompt = (
        f"{system_prompt}\n"
        f"{diet_line}\n"
        f"Servings: {servings}\n"
        "If any listed ingredient conflicts with the dietary preference, substitute it appropriately.\n\n"
        f"Ingredients: {ing_text}\n"
        f"Cuisine: {cuisine}\n"
        "Recipe:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    if decoding_strategy == "Beam Search":
        num_beams = 5
        do_sample = False
    else:
        num_beams = 1

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        repetition_penalty=1.2,
        num_return_sequences=1,
        do_sample=bool(do_sample),
        num_beams=int(num_beams),
        pad_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs.update(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Recipe:" in decoded:
        decoded = decoded.split("Recipe:", 1)[-1].strip()
    return decoded.strip()

#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")

# Task 2: system prompt style
prompt_choice = st.selectbox("🧩 Task 2: Prompt Style", list(SYSTEM_PROMPTS.keys()), index=0)

# Task 5 controls
dietary_pref = st.selectbox("✨ Dietary preference (Task 5):", DIET_OPTIONS, index=0)
servings = st.slider("✨ Servings (Task 5):", 1, 10, 2, 1)

# Task 1 controls
with st.expander("⚙️ Task 1: Text Generation Controls", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.selectbox("Temperature", [0.5, 1.0, 2.0], index=1)
        max_new_tokens = st.slider("Max new tokens", 100, 400, 250, step=25)
    with col2:
        do_sample = st.checkbox("Enable Sampling (do_sample)", value=True)
        top_k = st.selectbox("top_k", [5, 50], index=1)
    with col3:
        top_p = st.selectbox("top_p", [0.7, 0.95], index=1)
        decoding_strategy = st.radio("Decoding Strategy", ["Greedy", "Beam Search"], index=0, horizontal=True)

# Original inputs (kept)
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox(
    "Select a cuisine:",
    ["Any", "Asian", "Indian", "Middle Eastern", "Mexican", "Western", "Mediterranean", "African"]
)

# Generate recipe
if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["recipe"] = generate_recipe(
        ingredients=ingredients,
        cuisine=cuisine,
        system_prompt=SYSTEM_PROMPTS[prompt_choice],
        dietary_pref=dietary_pref,
        servings=servings,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        do_sample=bool(do_sample),
        decoding_strategy=decoding_strategy,
        max_new_tokens=int(max_new_tokens),
    )

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(
        label="📂 Save Recipe",
        data=st.session_state["recipe"],
        file_name="recipe.txt",
        mime="text/plain"
    )

    # Task 5: nutrition
    st.markdown("---")
    st.markdown("## 🧮 Nutrition Estimate (Task 5)")
    if st.button("Calculate Nutrition (Approx.)", use_container_width=True):
        with st.spinner("Estimating nutrition per serving..."):
            nutrition, raw = estimate_nutrition_llm(st.session_state["recipe"], servings=servings)

        if nutrition is not None:
            df = pd.DataFrame(
                [
                    ["Calories (kcal)", nutrition["calories_kcal"]],
                    ["Carbs (g)", nutrition["carbs_g"]],
                    ["Protein (g)", nutrition["protein_g"]],
                    ["Fat (g)", nutrition["fat_g"]],
                    ["Fiber (g)", nutrition["fiber_g"]],
                    ["Sugar (g)", nutrition["sugar_g"]],
                    ["Sodium (mg)", nutrition["sodium_mg"]],
                ],
                columns=["Nutrient (per serving)", "Estimated Amount"]
            )
            st.table(df)
            st.caption("Note: Values are approximate estimates generated from the recipe text.")
        else:
            st.warning("Could not parse nutrition output. Showing raw output below:")
            st.text_area("Raw nutrition output", raw, height=200)

   #  Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio(
        "Select Search Method:",
        ["Annoy (Fastest)", "Direct Search (Best Accuracy)"],
        index=0
    )

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        if search_method == "Annoy (Fastest)":
            alternatives = annoy_search_alternatives(ingredient_to_replace)
        else:
            alternatives = direct_search_alternatives(ingredient_to_replace)

        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
