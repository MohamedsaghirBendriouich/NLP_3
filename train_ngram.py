import os
import sys
import re
from collections import defaultdict, Counter
import random
import pickle

def tokenize(text):
    return re.findall(r'\w+', text)

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

def load_data(data_dir, max_files=None, max_sentences=None):
    sentences = []
    all_files = []

    # Try different data directories (for ease of use depending on the current working dir)
    possible_dirs = [
        data_dir,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    ]

    actual_dir = None
    for d in possible_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            actual_dir = d
            break

    if not actual_dir:
        print(f"Warning: Directory '{data_dir}' not found. Looked in: {possible_dirs}")
        return []

    print(f"Loading data from '{actual_dir}'...")

    for root, _, files in os.walk(actual_dir):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(root, f))

    # Shuffle files to get a diverse sample if limiting files
    random.seed(42)
    random.shuffle(all_files)

    if max_files:
        all_files = all_files[:max_files]

    for filepath in all_files:
        if max_sentences and len(sentences) >= max_sentences:
            break

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        tokens = tokenize(line)
                        if tokens:
                            sentences.append(tokens)
                            if max_sentences and len(sentences) >= max_sentences:
                                break
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return sentences

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'

    # Loading up to 500k sentences for a good balance of training speed and model quality
    sentences = load_data(data_dir, max_sentences=500000)

    if not sentences:
        print(f"Error: No sentences were loaded. Make sure the 'data' directory contains non-empty .txt files.")
        sys.exit(1)

    print(f"Loaded {len(sentences)} sentences.")

    n = 3
    print(f"Training {n}-gram model...")
    model = NGramModel(n)
    model.train(sentences)
    print("Training complete.")

    print("\nGenerating some sample texts:")
    for _ in range(5):
        print("-", model.generate())

    # Save in the same directory as the script
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'darija_{n}gram_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {save_path}")
