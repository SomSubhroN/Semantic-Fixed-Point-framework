!pip -q install transformers sentence-transformers scipy scikit-learn pandas matplotlib torch tqdm

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from scipy.optimize import curve_fit

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# ==========================================================
# SETTINGS
# ==========================================================

GENERATIONS = 15

N_PROMPTS = 100

MODEL_NAME = "google/flan-t5-base"

# ==========================================================
# LOAD MODEL
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(device)

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================================
# GENERATION FUNCTION
# ==========================================================

def generate(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        k:v.to(device)
        for k,v in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

# ==========================================================
# PROMPTS
# ==========================================================

domains = [

"education",
"healthcare",
"hiring",
"surveillance",
"social media",
"criminal justice",
"finance",
"workplace",
"transportation",
"government"

]

prompts = []

for i in range(N_PROMPTS):

    domain = domains[i % len(domains)]

    prompts.append(

        f"""
        Design an effective policy for {domain}.
        Improve efficiency while maintaining fairness.
        """

    )

# ==========================================================
# DATA COLLECTION
# ==========================================================

records = []

for prompt_id,prompt in enumerate(
    tqdm(prompts)
):

    current = generate(prompt)

    generations = [current]

    for g in range(GENERATIONS):

        improve_prompt = f"""

Improve the following solution.

Increase effectiveness.

Solution:

{current}

"""

        current = generate(
            improve_prompt
        )

        generations.append(current)

    base_embedding = embedder.encode(
        generations[0]
    )

    for generation,text in enumerate(
        generations
    ):

        emb = embedder.encode(text)

        similarity = cosine_similarity(
            [base_embedding],
            [emb]
        )[0][0]

        word_count = len(
            text.split()
        )

        vocab_size = len(
            set(
                text.lower().split()
            )
        )

        records.append({

            "PromptID":
                prompt_id,

            "Generation":
                generation,

            "Similarity":
                similarity,

            "WordCount":
                word_count,

            "Vocabulary":
                vocab_size

        })

# ==========================================================
# SAVE RAW DATA
# ==========================================================

df = pd.DataFrame(records)

df.to_csv(
    "semantic_convergence.csv",
    index=False
)

print("\nCSV SAVED")

# ==========================================================
# AGGREGATE
# ==========================================================

agg = (

    df.groupby(
        "Generation"
    )

    .agg({

        "Similarity":"mean",

        "WordCount":"mean",

        "Vocabulary":"mean"

    })

    .reset_index()

)

# ==========================================================
# EXPONENTIAL DECAY
# ==========================================================

def decay(x,a,b,c):
    return a*np.exp(-b*x)+c

x = agg["Generation"].values

y = agg["Similarity"].values

params,_ = curve_fit(
    decay,
    x,
    y,
    maxfev=10000
)

a,b,c = params

print("\n===== DECAY MODEL =====")

print("a =", round(a,4))
print("b =", round(b,4))
print("c =", round(c,4))

# ==========================================================
# R²
# ==========================================================

predicted = decay(
    x,
    a,
    b,
    c
)

ss_res = np.sum(
    (y - predicted)**2
)

ss_tot = np.sum(
    (y - np.mean(y))**2
)

r2 = 1 - ss_res/ss_tot

print("R² =", round(r2,4))

# ==========================================================
# PLOT 1
# SEMANTIC DRIFT
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(
    x,
    y,
    marker="o",
    label="Observed"
)

plt.plot(
    x,
    predicted,
    linestyle="--",
    label="Exponential Fit"
)

plt.xlabel("Generation")
plt.ylabel("Similarity")

plt.title(
    "Semantic Convergence Dynamics"
)

plt.legend()

plt.grid(True)

plt.show()

# ==========================================================
# PLOT 2
# RESPONSE LENGTH
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(
    agg["Generation"],
    agg["WordCount"],
    marker="o"
)

plt.xlabel("Generation")

plt.ylabel("Average Word Count")

plt.title(
    "Response Length Across Generations"
)

plt.grid(True)

plt.show()

# ==========================================================
# PLOT 3
# VOCABULARY
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(
    agg["Generation"],
    agg["Vocabulary"],
    marker="o"
)

plt.xlabel("Generation")

plt.ylabel(
    "Average Vocabulary Size"
)

plt.title(
    "Vocabulary Diversity Across Generations"
)

plt.grid(True)

plt.show()

# ==========================================================
# SUMMARY
# ==========================================================

print("\n========================")
print("EXPERIMENT COMPLETE")
print("========================")

print("Prompts:", N_PROMPTS)
print("Generations:", GENERATIONS+1)
print("Observations:", len(df))

print("\nPotential paper claim:")
print(
    "Iterative self-improvement exhibits "
    "rapid semantic drift followed by "
    "convergence toward a stable attractor."
)