import pandas as pd

print("Loading dataset...")

df = pd.read_csv("train.csv")

print("Dataset loaded")

# combine labels
df["label"] = (
    df["toxic"]
    + df["severe_toxic"]
    + df["insult"]
    + df["identity_hate"]
)

df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)

# keep only needed columns
df = df[["comment_text", "label"]]
df = df.rename(columns={"comment_text": "text"})

# balance dataset
safe = df[df["label"] == 0].sample(10000, random_state=42)
toxic = df[df["label"] == 1].sample(10000, random_state=42)

final_df = pd.concat([safe, toxic])

print("Saving dataset...")

final_df.to_csv("dataset.csv", index=False)

print("Done!")
print("Dataset size:", len(final_df))