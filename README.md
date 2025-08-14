# Universal Sentence Encoder Trainer

This project provides a complete pipeline to train a high-quality sentence embedding model from scratch on any language. 

## Motivation

Many state-of-the-art NLP models rely on pre-trained weights from large-scale models like BERT or T5. However, these models are not available for all languages, especially those that are less common or low-resource. 

This project was created to solve that problem. It allows you to build a powerful sentence encoder using a Transformer-based architecture, learning entirely from your own text data. The end result is a model that can produce meaningful vector representations (embeddings) of sentences, which can be used for tasks like semantic search, clustering, and sentence similarity in any language.

## Core Technologies

This project is built on a modern, state-of-the-art stack for NLP tasks, chosen for its power, flexibility, and strong community support.

- **Python**: The universal language for machine learning.
- **PyTorch**: A leading deep learning framework that provides flexibility and a dynamic computation graph, making it ideal for complex models and research.
- **Transformer Encoder Architecture**: The model is based on the same fundamental building block used in groundbreaking models like BERT and GPT. This is the de-facto standard for high performance in NLP.
- **Hugging Face `tokenizers`**: We use this library to train a custom, fast, and efficient Byte-Pair Encoding (BPE) tokenizer directly on your data.
- **Optuna**: A powerful, state-of-the-art hyperparameter optimization framework. It automates the tedious process of finding the best model architecture and learning parameters, using efficient sampling and pruning algorithms.
- **YAML**: All configuration is managed through a single, human-readable `config.yaml` file, separating the logic of the code from the experimental parameters.
- **Debiased Contrastive Loss**: In addition to the standard InfoNCE loss, this project supports Debiased Contrastive Loss. This advanced loss function accounts for sampling bias in contrastive learning, which can lead to more robust and higher-quality sentence embeddings, especially when dealing with datasets where false negatives might be a problem.

## Workflow Overview

The project supports a full end-to-end workflow:

1.  **Data Preparation**: Automatically splits your raw text into training and test sets.
2.  **Tokenizer Training**: Trains a custom tokenizer on your data.
3.  **Hyperparameter Optimization (Optional but Recommended)**: Uses Optuna to find the best model architecture (e.g., number of layers, embedding size) and training parameters for your specific dataset.
4.  **Pre-training**: The model first learns the structure of the language using a **masked and permuted language modeling** architecture.
5.  **Fine-tuning**: The pre-trained model is then fine-tuned using a contrastive learning objective. This project supports both the standard **InfoNCE loss** and a more advanced **Debiased Contrastive Loss**. This step is crucial for producing high-quality sentence embeddings that are well-suited for semantic similarity tasks.

## Installation and Usage

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. Create a Python virtual environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install the required packages
pip install -r requirements.txt
```

### How to Use

Follow these steps to train your own model.

**Step 1: Prepare Your Data**

Place your raw text data into a single file named `input.txt` in the root of the project directory. The file should have **one sentence per line**.

**Step 2: Initial Setup**

First, you need to process your data and create a tokenizer. Run the following commands in order:

```bash
# This splits your input.txt into training and test sets
python train.py --setup_data

# This trains a new tokenizer on your training data
python train.py --setup_tokenizer
```

**Step 3: Train the Model**

The training process happens in three stages.

**Step 3a: Find the Best Hyperparameters (Optional but Recommended)**

This first step uses Optuna to automatically find the best model configuration for your data. This can take a long time, but typically yields the best results.

```bash
# This will run the optimization process
python train.py --optimize
```

**Step 3b: Pre-train the Model**

Once the optimization is complete (or if you choose to skip it), you pre-train the model.

```bash
# This will pre-train the model
python train.py
```

By default, this command will automatically load the **best hyperparameters** found by the `--optimize` run. If you want to skip the optimization step and use the specific parameters defined in `config.yaml`, set `override_with_custom_params.enabled` to `true` in the `train` section of the config.

**Step 3c: Fine-tune the Model**

After the pre-training stage is complete, you must fine-tune the model to produce high-quality sentence embeddings. This step is crucial for performance on downstream tasks.

```bash
# This will fine-tune the pre-trained model
python train.py --finetune
```

### Starting Fresh (`--new` flag)

By default, the script will always try to resume from the last saved state. If you want to start a process from scratch, use the `--new` flag.

-   `python train.py --new`: Ignores any saved pre-trained model and starts training from scratch.
-   `python train.py --optimize --new`: Deletes the old optimization study and starts a new one.
-   `python train.py --finetune --new`: Ignores any previously fine-tuned model and starts a new fine-tuning session from the pre-trained model.

## Configuration (`config.yaml`)

The `config.yaml` file is the central control panel for the entire project. Here are some of the most important settings:

-   `data.test_size`: The proportion (e.g., `0.1` for 10%) of your dataset to hold out for testing.
-   `tokenizer.vocab_size`: The total number of unique tokens you want in your tokenizer's vocabulary. A larger size captures more words but increases model size.
-   `tokenizer.keep_punctuation`: A boolean (`true` or `false`) to control whether punctuation is stripped from the text or kept as separate tokens.
-   `optimize.n_trials`: The total number of different hyperparameter combinations Optuna should test.
-   `train.override_with_custom_params.enabled`: Set this to `true` to ignore the results from Optuna and instead use the specific model parameters defined under this section for pre-training.
-   `finetune.freeze_layer_ratio`: A float between 0.0 and 1.0. This controls what percentage of the model's **bottom layers** are frozen during fine-tuning. For example, `0.25` freezes the first 25% of the layers, preserving their general knowledge.
-   `finetune.scheduler_patience`: The number of epochs with no improvement on the validation loss before the learning rate is reduced.

## Final Output

-   **`models/`**: This directory contains intermediate model weights from the pre-training (`model.pt`) and fine-tuning (`model_finetuned.pt`) steps. These are useful for resuming training.
-   **`out/`**: This directory contains the final, production-ready artifacts:
    -   `model.pt`: A self-contained, easy-to-use `SentenceEncoder` model that directly outputs sentence embeddings.
    -   `tokenizer.json`: The exact tokenizer that was trained and used with the final model.

## Using the Final Model for Inference

The fully-wrapped, production-ready model and its tokenizer are saved to the `out/` directory after the fine-tuning process is complete.

Here is an example of how to load the model and use it to encode sentences. You can save this as a separate Python script (e.g., `inference_example.py`):

```python
import torch
from tokenizers import Tokenizer
import os

# --- 1. Define paths and check for model/tokenizer ---
def get_sentence_embedding(sentences, model_dir='out'):
    """
    Loads the final model and tokenizer from the output directory to generate embeddings.
    """
    model_path = os.path.join(model_dir, 'model.pt')
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Error: Model or tokenizer not found in '{model_dir}'.")
        print("Please run the full training and fine-tuning pipeline first using 'python train.py --finetune'.")
        return

    try:
        # --- 2. Load the saved model and tokenizer ---
        # Note: We use torch.load() on the model file directly because we saved the entire model object.
        # We also specify map_location='cpu' to ensure it works on machines without a GPU.
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("Model and tokenizer loaded successfully.")

    except Exception as e:
        print(f"An error occurred while loading the model or tokenizer: {e}")
        return

    # Set the model to evaluation mode
    model.eval()

    # --- 3. Tokenize the input sentences ---
    # The tokenizer needs to pad the sentences to the same length for batch processing.
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    encoded_input = tokenizer.encode_batch(sentences)

    # Convert tokenized inputs to PyTorch tensors
    input_ids = torch.tensor([e.ids for e in encoded_input], dtype=torch.long)
    attention_mask = torch.tensor([e.attention_mask for e in encoded_input], dtype=torch.bool)

    # --- 4. Generate embeddings ---
    # We don't need to calculate gradients for inference.
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)

    return embeddings

if __name__ == '__main__':
    # List of sentences you want to encode
    my_sentences = [
        "This is an example sentence.",
        "Each sentence is converted to a vector.",
        "This is a great way to represent text for machine learning."
    ]

    sentence_embeddings = get_sentence_embedding(my_sentences)

    if sentence_embeddings is not None:
        print("\n--- Results ---")
        print(f"Shape of the embedding tensor: {sentence_embeddings.shape}")
        print("(Batch Size, Embedding Dimension)")
        
        print("\nEmbedding for the first sentence:")
        print(sentence_embeddings[0])
```


## Semantic Search with `semantic_search.py`

This project includes a powerful script, `semantic_search.py`, that allows you to perform semantic searches on a large text corpus using the final trained model. It efficiently handles large datasets by generating and searching through embedding chunks.

### How It Works

1.  **Embedding Generation**: The first time you run the script, it will encode all sentences from your `input.txt` file and save them into a `data/embedding_chunks/` directory. This is a one-time process.
2.  **Efficient Searching**: On subsequent runs, the script will load these pre-computed embedding chunks, allowing for very fast semantic searches without needing to re-encode the entire dataset each time.

### How to Use

To perform a semantic search, run the script from your terminal with a query sentence:

```bash
python semantic_search.py "Your search query goes here"
```

**Example:**

```bash
python semantic_search.py "What are the main themes in this text?"
```

### Options

-   `--top_k <number>`:  Specify the number of top results to return. The default is `30`. 
    
    ```bash
    python semantic_search.py "A question about the text" --top_k 5
    ```

-   `--new`: Force the script to regenerate the embedding chunks. Use this if your model has changed, you have updated your `input.txt` file or want to start fresh.

    ```bash
    python semantic_search.py "A new search query" --new
    ```

## Case Study: Shakespearean Language Model

This case study analyzes the performance of a sentence-embedding model trained on Shakespearean English. The model's ability to understand and match reformulated queries with their original counterparts is evaluated.

### Model Training and Optimization

The model's architecture was optimized using Optuna over 50 iterations, with each iteration consisting of 5 epochs and 200 batches per epoch. After optimization, the model was trained for 24 epochs and then fine-tuned for 16 epochs using the default configuration.

The training dataset consists of 105,591 sentences, and the test dataset consists of 11,733 sentences.

### Multilingual Potential

Although this model was trained on English (specifically, the works of Shakespeare), the underlying architecture and training process are language-agnostic. Because the model and tokenizer are trained from scratch on the provided text, this pipeline can be used for any language with a similar expectation of performance, provided a sufficiently large and clean dataset.

### Search Results

The following examples demonstrate the model's performance in matching reformulated queries to their original sentences.

### Example 1

*   **Original:** `But could be willing to march on to Calais`
*   **Query:** `Yet, I would be willing to march on to Calais.`
*   **Rank:** 1

1.  **`But could be willing to march on to Calais`**
2.  `the dry fool drink, then is the fool not dry; bid the dishonest man`
3.  `When were you wont to be so full of songs, sirrah?`
4.  `To say they err, I dare not be so bold,`
5.  `Which cannot look more hideously upon me`

### Example 2

*   **Original:** `In some of our best ports, and are at point`
*   **Query:** `In some of our finest ports, and stand at point.`
*   **Rank:** 1

1.  **`In some of our best ports, and are at point`**
2.  `knave very voluble; no further conscionable than in putting on the mere`
3.  `Plainly conceive, I love you.`
4.  `lady worse.`
5.  `’Has broke my head across, and has given Sir Toby a bloody coxcomb too.`

### Example 3

*   **Original:** `Sun, hide thy beams, Timon hath done his reign.`
*   **Query:** `Sun, conceal thy beams, for Timon's reign is done.`
*   **Rank:** 1

1.  **`Sun, hide thy beams, Timon hath done his reign.`**
2.  `O had thy mother borne so hard a mind,`
3.  `[_Offers the letter again._]`
4.  `An humble heart.`
5.  `For both hast thou, and both become the grave.`

### Example 4

*   **Original:** `And I will speak, that so my heart may burst.`
*   **Query:** `And I shall speak, that so my heart may burst.`
*   **Rank:** 17

1.  `Should slip so grossly, both in the heat of blood`
2.  `Now I do frown on thee with all my heart,`
3.  `night. O sweet Pistol! Away, Bardolph!`
4.  `_ Not a flower, not a flower sweet,`
5.  `Now, Master Shallow, you’ll complain of me to the King?`

...17. **`And I will speak, that so my heart may burst.`**

### Example 5

*   **Original:** `Where in the purlieus of this forest stands`
*   **Query:** `Where, in the purlieus of this wood, doth stand?`
*   **Rank:** 46

1.  `Fight closer, or, good faith, you’ll catch a blow.`
2.  `Where in the purlieus of this forest stands`
3.  `Look what an unthrift in the world doth spend`
4.  `Then in our measure do but vouchsafe one change.`
5.  `Can you nominate in order now the degrees of the lie?`

...46. **`Where in the purlieus of this forest stands`**

### Example 6

*   **Original:** `The least of you shall share his part thereof.`
*   **Query:** `The least among you shall partake of his share.`
*   **Rank:** 230

1.  `I know you have determined to bestow her`
2.  `No more than I am well acquitted of.`
3.  `The article of your oath, which you shall never`
4.  `lion, that holds his pole-axe sitting on a close-stool, will be given`
5.  `Th’ offence is not of such a bloody nature,`

...230. **`The least of you shall share his part thereof.`**

### Example 7

*   **Original:** `[_Aside_.] That such an ass should owe them.`
*   **Query:** `That such a fool should be in their debt.`
*   **Rank:** 333

1.  `My good knave Costard, exceedingly well met.`
2.  `_Why should this a desert be?`
3.  `GREEN - Servant to King Richard`
4.  `[_Comes forward_.] Dumaine, thy love is far from charity,`
5.  `Juno sings her blessings on you._`

...333. **`[_Aside_.] That such an ass should owe them.`**

### Example 8

*   **Original:** `And prey on garbage.`
*   **Query:** `And feast upon that which the vultures leave.`
*   **Rank:** 12837

1.  `To let the wretched man outlive his wealth,`
2.  `never out of my bones. I shall not fear fly-blowing.`
3.  `Which hath our several honours all engag’d`
4.  `Of youth upon him, from which the world should note`
5.  `* * * * * *`

...12837. **`And prey on garbage.`**

## Further Evaluation

To further evaluate the model's performance, a new set of 10 examples was created. The original sentences were randomly selected from the `input.txt` corpus, and new queries were formulated in a Shakespearean style.

### Example 9

*   **Original:** `Look in thy glass and tell the face thou viewest,`
*   **Query:** `Gaze upon thy reflection, and spake to the visage thou dost see.`
*   **Rank:** 1

1.  **`Look in thy glass and tell the face thou viewest,`**
2.  `(Which still hath been both grave and prosperous)`
3.  `Can cunning sin cover itself withal.`
4.  `In charging you with matters, to commit you,`
5.  `When thou wast here above the ground, I was`

### Example 10

*   **Original:** `And see thy blood warm when thou feel’st it cold.`
*   **Query:** `And feel thy blood course warm when thou art cold.`
*   **Rank:** 1

1.  **`And see thy blood warm when thou feel’st it cold.`**
2.  `I, that am rudely stamped, and want love’s majesty`
3.  `Thou rascal, that art worst in blood to run,`
4.  `I likewise hear that Valentine is dead.`
5.  `hyena, and that when thou are inclined to sleep.`

### Example 11

*   **Original:** `And your sweet semblance to some other give.`
*   **Query:** `And thy sweet likeness to another grant.`
*   **Rank:** 1

1.  **`And your sweet semblance to some other give.`**
2.  `My mother’s son, sir.`
3.  `That fled the snares of watchful tyranny;`
4.  `I’ll pray a thousand prayers for thy death,`
5.  `The time will not allow the compliment`

### Example 12

*   **Original:** `And for my self mine own worth do define,`
*   **Query:** `And for myself, my own value I shall ascertain.`
*   **Rank:** 1

1.  **`And for my self mine own worth do define,`**
2.  `As I not for my self, but for thee will,`
3.  `I leave an arrant knave with your worship; which I beseech your`
4.  `I pluck this white rose with Plantagenet.`
5.  `to no more payments, fear no more tavern bills, which are often the`

### Example 13

*   **Original:** `My body is the frame wherein ’tis held,`
*   **Query:** `My form is the vessel in which it is contained.`
*   **Rank:** 1

1.  **`My body is the frame wherein ’tis held,`**
2.  `Suggest his soon-believing adversaries,`
3.  `With well-appointed powers. He is a man`
4.  `I would learn that; for by the marks of sovereignty, knowledge and`
5.  `Esteem none friends but such as are his friends,`

### Example 14

*   **Original:** `For having traffic with thyself alone,`
*   **Query:** `For conducting commerce with naught but thyself,`
*   **Rank:** 49

1.  `Why, look where he comes; and my good man too. He’s as far from`
2.  `Till seven at night; to make society`
3.  `I hope I shall know your honour better.`
4.  `Shall I? O rare! By the Lord, I’ll be a brave judge.`
5.  `At last it rains, and busy winds give o’er.`

...49. **`For having traffic with thyself alone,`**

### Example 15

*   **Original:** `Making a famine where abundance lies,`
*   **Query:** `Creating a dearth where plenty doth reside,`
*   **Rank:** 1211

1.  `Where beauty’s veil doth cover every blot,`
2.  `Aumerle._]`
3.  `Convey him hence, and on our longboat’s side`
4.  `Enter some, bringing in the Clerk of Chartham.`
5.  `First, tell me, have you ever been at Pisa?`

...1211. **`Making a famine where abundance lies,`**

### Example 16

*   **Original:** `Pity the world, or else this glutton be,`
*   **Query:** `Have pity on the world, lest thou be a glutton.`
*   **Rank:** 1497

1.  `Marcus, even thou hast struck upon my crest,`
2.  `Agamemnon—how if he had boils, full, all over, generally?`
3.  `No pray thee, let it serve for table-talk.`
4.  `Nothing but “Willow, willow, willow,” and between`
5.  `You are betroth’d both to a maid and man.`

...1497. **`Pity the world, or else this glutton be,`**

### Example 17

*   **Original:** `And summer’s green all girded up in sheaves`
*   **Query:** `And summer's verdure all bound up in bundles,`
*   **Rank:** 1645

1.  **`And summer’s green all girded up in sheaves`**
2.  `The name of valour. [_Sees his dead father_.] O, let the vile world end`
3.  `Away with her to prison. Go to, no more words.`
4.  `Are all thrown down, and that which here stands up`
5.  `Shall be the surety for their traitor father.`

...1645. **`And summer’s green all girded up in sheaves`**

### Example 18

*   **Original:** `His tender heir might bear his memory:`
*   **Query:** `His gentle heir might carry on his name.`
*   **Rank:** 13819

1.  `yourself from whipping, leap me over this stool and run away.`
2.  `You fur your gloves with reason. Here are your reasons:`
3.  `Good day, my lord.`
4.  `Perpetual durance?`
5.  `And there I left him tranc’d.`

...13819. **`His tender heir might bear his memory:`**

### Analysis of Further Evaluation

This second set of examples provides further insight into the model's capabilities. The model continues to perform well when the query is a close paraphrase of the original sentence. However, it struggles with more abstract or metaphorical reformulations.

For instance, the query "His gentle heir might carry on his name" for the original "His tender heir might bear his memory:" results in a very low rank. This suggests that the model has not fully grasped the metaphorical connection between "bearing memory" and "carrying on a name."

These results reinforce the initial analysis: the model is a powerful tool for semantic search, but its understanding of language is more literal than abstract. Further training on a more diverse and abstract dataset could help to improve its performance on these more challenging queries.
