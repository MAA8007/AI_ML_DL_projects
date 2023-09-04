#Data Science Job Posting on Glassdoor

from transformers import pipeline
import pandas as pd

df = pd.read_csv("job/Uncleaned_DS_jobs.csv")
job_descriptions = df['Job Description'].to_numpy()

print(f)