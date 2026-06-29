!pip -q install transformers sentence-transformers scipy scikit-learn pandas matplotlib torch tqdm

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import f_oneway

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# =====================================================
# SETTINGS
# =====================================================

MODEL_NAME = "google/flan-t5-small"

GENERATIONS = 15

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
# 50 DOMAIN-TAGGED PROMPTS
# =====================================================

prompts = [

("education","Design an effective policy for online education while maintaining fairness."),
("education","Develop a framework for AI-assisted student assessment."),
("education","Create a policy for detecting academic misconduct."),
("education","Design a system for personalized learning recommendations."),
("education","Develop a policy for educational resource allocation."),

("healthcare","Design an effective healthcare triage policy."),
("healthcare","Develop a framework for AI-assisted medical diagnosis."),
("healthcare","Create a policy for allocating scarce medical resources."),
("healthcare","Design a patient monitoring system that balances efficiency and fairness."),
("healthcare","Develop a healthcare decision-support framework."),

("hiring","Design an effective employee recruitment policy."),
("hiring","Develop a framework for AI-assisted hiring decisions."),
("hiring","Create a fair candidate screening process."),
("hiring","Design a workforce talent identification system."),
("hiring","Develop a policy for employee promotion decisions."),

("surveillance","Design a public surveillance governance policy."),
("surveillance","Develop a framework for AI-based threat detection."),
("surveillance","Create a policy for balancing surveillance and privacy."),
("surveillance","Design an urban security monitoring system."),
("surveillance","Develop a framework for automated incident detection."),

("social_media","Design a content moderation policy for social media."),
("social_media","Develop a misinformation detection framework."),
("social_media","Create a policy for combating online harassment."),
("social_media","Design a recommendation system governance framework."),
("social_media","Develop a strategy for managing harmful online content."),

("criminal_justice","Design an AI-assisted criminal risk assessment policy."),
("criminal_justice","Develop a framework for judicial decision support."),
("criminal_justice","Create a fair parole recommendation system."),
("criminal_justice","Design a policy for predictive policing technologies."),
("criminal_justice","Develop a framework for evidence evaluation support."),

("finance","Design a financial fraud detection policy."),
("finance","Develop an AI-assisted loan approval framework."),
("finance","Create a policy for automated credit risk assessment."),
("finance","Design a system for anti-money laundering monitoring."),
("finance","Develop a framework for financial compliance monitoring."),

("workplace","Design a workplace productivity optimization policy."),
("workplace","Develop a framework for employee performance evaluation."),
("workplace","Create a policy for workplace resource allocation."),
("workplace","Design a workforce scheduling optimization system."),
("workplace","Develop a framework for organizational decision support."),

("transportation","Design a smart traffic management policy."),
("transportation","Develop a framework for autonomous transportation governance."),
("transportation","Create a policy for urban mobility optimization."),
("transportation","Design a public transport resource allocation system."),
("transportation","Develop a framework for traffic incident response."),

("government","Design a public service delivery optimization policy."),
("government","Develop a framework for government resource allocation."),
("government","Create a policy for digital governance systems."),
("government","Design a citizen grievance management framework."),
("government","Develop a strategy for public policy decision support.")

]

# =====================================================
# EXPERIMENT
# =====================================================

results = []

for prompt_id,(domain,prompt) in enumerate(
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

        r2 = 1 - (ss_res/ss_tot)

        if r2 < 0.50:
            continue

        results.append({

            "PromptID": prompt_id,
            "Domain": domain,
            "a": a,
            "b": b,
            "c": c,
            "r2": r2

        })

    except:
        continue

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame(results)

df.to_csv(
    "domain_convergence_study.csv",
    index=False
)

print("\nCSV SAVED")

# =====================================================
# OVERALL SUMMARY
# =====================================================

mean_c = df["c"].mean()
mean_b = df["b"].mean()
mean_r2 = df["r2"].mean()

std_c = df["c"].std()

n = len(df)

ci = (
    1.96 *
    std_c /
    np.sqrt(n)
)

print("\n====================")
print("OVERALL RESULTS")
print("====================")

print("Valid Fits =", n)
print("Mean c =", round(mean_c,4))
print("95% CI = ±", round(ci,4))
print("Mean b =", round(mean_b,4))
print("Mean R² =", round(mean_r2,4))

# =====================================================
# DOMAIN SUMMARY
# =====================================================

domain_summary = (

    df.groupby("Domain")

    .agg({

        "c":["mean","std","count"],

        "b":["mean","std"],

        "r2":["mean"]

    })

)

print("\n====================")
print("DOMAIN SUMMARY")
print("====================")

print(domain_summary)

# =====================================================
# DOMAIN ANOVA
# =====================================================

groups = []

for domain in df["Domain"].unique():

    groups.append(

        df[
            df["Domain"] == domain
        ]["c"]

    )

anova = f_oneway(*groups)

print("\n====================")
print("DOMAIN ANOVA")
print("====================")

print("F =", anova.statistic)
print("p =", anova.pvalue)

# =====================================================
# DOMAIN BOXPLOT
# =====================================================

plt.figure(figsize=(12,6))

df.boxplot(
    column="c",
    by="Domain",
    rot=45
)

plt.title(
    "Domain-wise Semantic Attractors"
)

plt.suptitle("")

plt.ylabel(
    "Asymptote c"
)

plt.show()

# =====================================================
# DOMAIN BARPLOT
# =====================================================

means = df.groupby(
    "Domain"
)["c"].mean()

stds = df.groupby(
    "Domain"
)["c"].std()

plt.figure(figsize=(12,6))

plt.bar(
    means.index,
    means.values,
    yerr=stds.values
)

plt.xticks(
    rotation=45
)

plt.ylabel(
    "Mean Attractor c"
)

plt.title(
    "Mean Semantic Attractor by Domain"
)

plt.show()

# =====================================================
# SAVE DOMAIN SUMMARY
# =====================================================

domain_summary.to_csv(
    "domain_summary.csv"
)

print("\n================================")
print("DOMAIN STUDY COMPLETE")
print("================================")

print("\nFiles Generated:")
print("1. domain_convergence_study.csv")
print("2. domain_summary.csv")