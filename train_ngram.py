import os
import re
from collections import defaultdict, Counter
import random
import pickle

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.model = defaultdict(dict)

    def train(self, sentences):
        for sentence in sentences:
            padded = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                word = padded[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        for context, counter in self.ngram_counts.items():
            total = self.context_counts[context]
            for word, count in counter.items():
                self.model[context][word] = count / total

    def generate(self, max_length=50):
        context = tuple(['<s>'] * (self.n - 1))
        result = []
        for _ in range(max_length):
            if context not in self.model:
                break
            choices = list(self.model[context].keys())
            probs = list(self.model[context].values())
            word = random.choices(choices, weights=probs)[0]
            if word == '</s>':
                break
            result.append(word)
            context = tuple(list(context)[1:] + [word])
        return ' '.join(result)

def load_data(data_dir, max_files=10, lines_per_file=10000):
    sentences = []
    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(root, f))

    random.shuffle(all_files)

    for filepath in all_files[:max_files]:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                random.shuffle(lines)
                for line in lines[:lines_per_file]:
                    line = line.strip()
                    if line:
                        tokens = tokenize(line)
                        if tokens:
                            sentences.append(tokens)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return sentences

if __name__ == '__main__':
    print("Loading data...")
    sentences = load_data('data', max_files=15, lines_per_file=20000)
    print(f"Loaded {len(sentences)} sentences.")

    n = 3
    print(f"Training {n}-gram model...")
    model = NGramModel(n)
    model.train(sentences)
    print("Training complete.")

    print("\nGenerating some sample texts:")
    for _ in range(5):
        print("-", model.generate())

    with open(f'darija_{n}gram_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to darija_{n}gram_model.pkl")
