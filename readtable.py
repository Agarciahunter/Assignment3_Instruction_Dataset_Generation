from datasets import load_dataset
import numpy as np
# Load the dataset
dataset = load_dataset("jahjinx/IMDb_movie_reviews")

# Access the 'train' split
train_data = dataset["train"]

# Count the number of words in the 'text' column
word_count = sum(len(review.split()) for review in train_data["text"])

print("Total number of words in the 'text' column:", word_count)

# Calculate the word count for each review
word_counts = [len(review.split()) for review in train_data["text"]]

# Calculate the maximum, minimum, average, and median word counts
max_word_count = max(word_counts)
min_word_count = min(word_counts)
average_word_count = np.mean(word_counts)
median_word_count = np.median(word_counts)

print("Maximum word count:", max_word_count)
print("Minimum word count:", min_word_count)
print("Average word count:", average_word_count)
print("Median word count:", median_word_count)