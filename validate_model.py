import pickle
from train_ngram import NGramModel

def main():
    print("Loading model...")
    with open('darija_3gram_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    print("\nGenerating validation texts:")
    for i in range(10):
        print(f"{i+1}. {model.generate()}")

if __name__ == "__main__":
    main()
