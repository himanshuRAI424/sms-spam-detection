# SMS Spam Detection using LSTM

This project implements an **SMS Spam Detection system using Deep Learning and Natural Language Processing (NLP)**. The model classifies SMS messages into two categories: **Spam** and **Ham (non-spam)**.

The project uses a **Long Short-Term Memory (LSTM)** neural network to learn patterns from SMS text and predict whether a new message is spam.

## Project Workflow

```text
SMS Dataset
     ↓
Load Dataset
     ↓
Clean Dataset
     ↓
Remove Duplicate Rows
     ↓
Convert Labels
(Ham = 0, Spam = 1)
     ↓
Train-Test Split
     ↓
Text Tokenization
     ↓
Convert Text to Sequences
     ↓
Padding
     ↓
Embedding Layer
     ↓
LSTM Layer
     ↓
Dropout
     ↓
Sigmoid Output
     ↓
Spam / Ham Prediction
```

## Dataset

The project uses the `spam.csv` dataset.

The dataset contains:

* `v1` → Message label
* `v2` → SMS message

The labels are converted as:

```text
ham  → 0
spam → 1
```

Duplicate rows are removed before training.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras
* Scikit-learn
* Seaborn
* Natural Language Processing (NLP)
* LSTM

## Data Preprocessing

The following preprocessing steps are performed:

1. Load the `spam.csv` dataset.
2. Select the required `v1` and `v2` columns.
3. Rename them to `label` and `message`.
4. Check missing values and duplicate rows.
5. Remove duplicate messages.
6. Convert `ham` and `spam` labels into `0` and `1`.
7. Split the dataset into training and testing data.

The dataset is divided using an **80:20 train-test split** with stratification.

## Text Processing

The text messages are converted into numerical sequences using Keras `Tokenizer`.

```python
Tokenizer(num_words=5000, oov_token="<OOV>")
```

The sequences are then padded to a fixed length of:

```text
MAX_LEN = 100
```

This ensures that all messages have the same input length.

## Deep Learning Model

The project uses the following neural network architecture:

```text
Embedding Layer
      ↓
LSTM (64 Units)
      ↓
Dropout (0.3)
      ↓
Dense Layer (1 Unit)
      ↓
Sigmoid Activation
```

### Model Configuration

* Vocabulary size: `5000`
* Embedding dimension: `64`
* Maximum sequence length: `100`
* LSTM units: `64`
* Dropout: `0.3`
* Output activation: `Sigmoid`
* Optimizer: `Adam`
* Loss function: `Binary Crossentropy`
* Batch size: `32`
* Maximum epochs: `30`

## Early Stopping

Early stopping is used during training to prevent unnecessary training when validation loss stops improving.

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

## Model Evaluation

The trained model is evaluated on the test dataset using:

* Test Loss
* Test Accuracy
* Predicted Probabilities
* Confusion Matrix
* Training vs Validation Accuracy Graph

The confusion matrix shows the classification results for:

```text
Ham
Spam
```

## Predicting a New SMS

The project also allows the user to enter a new SMS message.

The message goes through the same tokenization and padding process before being passed to the trained LSTM model.

Example:

```text
Enter your SMS: Congratulations! You have won a free prize.
```

The model then produces a prediction:

```text
Prediction: SPAM
```

For a non-spam message, the project displays:

```text
Prediction: correct information
```

## Project Structure

```text
sms-spam-detection-lstm/
│
├── spam.csv
├── mail_spam_detection.ipynb
└── README.md
```

## How to Run

Install the required libraries:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn seaborn
```

Then open the notebook:

```text
mail_spam_detection.ipynb
```

Run the cells step by step to:

1. Load the dataset
2. Preprocess the SMS messages
3. Train the LSTM model
4. Evaluate the model
5. Test new SMS messages

## Key Learning Outcomes

This project demonstrates practical implementation of:

* Natural Language Processing
* Text Tokenization
* Sequence Padding
* Word Embeddings
* LSTM Networks
* Binary Classification
* Model Evaluation
* Confusion Matrix
* Early Stopping
* Real-time SMS Prediction

## Author

**Himanshu Choudhary**
