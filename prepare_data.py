import pandas as pd

# Load datasets
fake = pd.read_csv("C:\\Users\\sneha\\OneDrive\\Attachments\\Desktop\\trellisoft.ai\\fake_news_detector\\Fake.csv\\Fake.csv")
real = pd.read_csv("C:\\Users\\sneha\\OneDrive\\Attachments\\Desktop\\trellisoft.ai\\fake_news_detector\\True.csv\\True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine both
df = pd.concat([fake, real])

# Keep only text + label
df = df[["text", "label"]]

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# Save new dataset
df.to_csv("data.csv", index=False)

print("✅ data.csv created successfully")