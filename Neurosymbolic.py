#NeuroSymbolic AI - George Wagenknecht - 2023

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
def preprocess(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    preprocessed_text = " ".join(words)

    return preprocessed_text

# Preprocess the text stimuli
text1 = "This is the first text."#environment stimuli
text2 = "This is the second text."#cognition (including associative qualia map, associative reasoning, i.e learning)
preprocessed_text1 = preprocess(text1)
preprocessed_text2 = preprocess(text2)

# Vectorize the text stimuli using the BoW model
vectorizer = CountVectorizer()
vectorizer.fit([preprocessed_text1, preprocessed_text2])
vectorized_text1 = vectorizer.transform([preprocessed_text1]).toarray()
vectorized_text2 = vectorizer.transform([preprocessed_text2]).toarray()

# Add Gaussian noise to the vectorized arrays
noise_scale = 0.1 # adjust this value to control the amount of noise #TODO: create 5D noise map, simmilar to electrical signalling in the brain and add a qualia dimension

noise_array1 = np.random.normal(scale=noise_scale, size=vectorized_text1.shape)
noise_array2 = np.random.normal(scale=noise_scale, size=vectorized_text2.shape)
vectorized_text1 = vectorized_text1 + noise_array1
vectorized_text2 = vectorized_text2 + noise_array2

# Calculate the cosine similarity between the two text stimuli
similarity_score = cosine_similarity(vectorized_text1, vectorized_text2)

# Perform associative reasoning
if similarity_score > 0.3:
    print("The two text stimuli are related.")
else:
    print("The two text stimuli are not related.")
#implement cybernetic method