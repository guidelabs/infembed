def run(dataset_path, vocab_size, save_path):
    from tinystories import tinystories_tokenizer_raw, TinyStoriesDataset

    raw_tokenizer = tinystories_tokenizer_raw()
    dataset = TinyStoriesDataset(dataset_path)
    tokenizer = raw_tokenizer.train_new_from_iterator(dataset, vocab_size)
    tokenizer.save_pretrained(save_path)


DATASET_PATH = (
    "/home/ubuntu/Documents/infembed/files/tinystories/TinyStoriesV2-GPT4-train.txt"
)
VOCAB_SIZE = 10000
SAVE_PATH = "/home/ubuntu/Documents/infembed/data/_core/tinystories_tokenizer"


if __name__ == "__main__":
    run(DATASET_PATH, VOCAB_SIZE, SAVE_PATH)
