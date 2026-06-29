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

# =====================================================
# SETTINGS
# =====================================================

MODEL_NAME = "google/flan-t5-small"

GENERATIONS = 15
N_PROMPTS = 50

OBJECTIVE = "Improve effectiveness"

# =====================================================
# LOAD MODEL
# =====================================================

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

# =====================================================
# GENERATION FUNCTION
# =====================================================

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

# =====================================================
# EXPONENTIAL MODEL
# =====================================================

def decay(x,a,b,c):
    return a*np.exp(-b*x)+c

# =====================================================
# PROMPTS
# =====================================================

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

# =====================================================
# EXPERIMENT
# =====================================================

results = []

for prompt_id,prompt in enumerate(
    tqdm(prompts)
):

    current = generate(prompt)

    generations = [current]

    for g in range(GENERATIONS):

        improve_prompt = f"""
Improve the following solution.

{OBJECTIVE}

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

    similarities = []

    for text in generations:

        emb = embedder.encode(text)

        sim = cosine_similarity(
            [base_embedding],
            [emb]
        )[0][0]

        similarities.append(sim)

    x = np.arange(
        len(similarities)
    )

    y = np.array(
        similarities
    )

    try:

        params,_ = curve_fit(

            decay,

            x,

            y,

            bounds=(
                [0,0,-1],
                [2,5,1]
            ),

            maxfev=10000
        )

        a,b,c = params

        predicted = decay(
            x,
            a,
            b,
            c
        )

        ss_res = np.sum(
            (y-predicted)**2
        )

        ss_tot = np.sum(
            (y-np.mean(y))**2
        )

        if ss_tot == 0:
            continue

        r2 = (
            1 - ss_res/ss_tot
        )

        if r2 < 0.70:
            continue

        results.append({

            "PromptID":
                prompt_id,

            "a":
                a,

            "b":
                b,

            "c":
                c,

            "r2":
                r2

        })

    except:
        continue

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame(results)

df.to_csv(
    "flan_t5_small_50_prompt_study.csv",
    index=False
)

print("\nCSV SAVED")

# =====================================================
# SUMMARY
# =====================================================

mean_c = df["c"].mean()
std_c = df["c"].std()

mean_b = df["b"].mean()
std_b = df["b"].std()

mean_r2 = df["r2"].mean()

n = len(df)

ci = (
    1.96 *
    std_c /
    np.sqrt(n)
)

print("\n==========================")
print("FINAL RESULTS")
print("==========================")

print("Valid Fits:", n)

print("\nMean c =", round(mean_c,4))
print("95% CI = ±", round(ci,4))

print("\nMean b =", round(mean_b,4))

print("\nMean R² =", round(mean_r2,4))

# =====================================================
# HISTOGRAM
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["c"],
    bins=15
)

plt.xlabel(
    "Asymptote c"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Semantic Attractors"
)

plt.grid(True)

plt.show()

# =====================================================
# BOXPLOT
# =====================================================

plt.figure(figsize=(6,6))

plt.boxplot(
    df["c"]
)

plt.ylabel(
    "Asymptote c"
)

plt.title(
    "Semantic Attractor Distribution"
)

plt.show()

# =====================================================
# SCATTER
# =====================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["b"],
    df["c"]
)

plt.xlabel(
    "Decay Rate (b)"
)

plt.ylabel(
    "Asymptote (c)"
)

plt.title(
    "Decay Rate vs Attractor"
)

plt.grid(True)

plt.show()

# =====================================================
# SAVE SUMMARY
# =====================================================

summary = pd.DataFrame({

    "Metric":[
        "Mean c",
        "CI",
        "Mean b",
        "Mean R2",
        "Valid Fits"
    ],

    "Value":[
        mean_c,
        ci,
        mean_b,
        mean_r2,
        n
    ]
})

summary.to_csv(
    "summary_statistics.csv",
    index=False
)

print("\n===================================")
print("EXPERIMENT COMPLETE")
print("===================================")

print("\nFiles Generated:")

print("1. flan_t5_small_50_prompt_study.csv")

print("2. summary_statistics.csv")