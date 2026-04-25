import os
import pickle
from train_ngram import NGramModel

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'darija_3gram_model.pkl')
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    print("\nGenerating validation texts:")
    for i in range(10):
        print(f"{i+1}. {model.generate()}")

if __name__ == "__main__":
    main()
