# GPT-2 Streamlit Sales bot

This repository contains a Streamlit-based web application that uses a fine-tuned GPT-2 model for text generation. The app allows users to enter a prompt and generate text based on it.

I still am trying to figure out what model I want to use for this, GPT-2 was the easiest to get something going. The idea is to have the bot trained to my habits/wording replies from (agency_mvp_text_threads.csv) ; this file is ~ 2 years of text threads between me and prospects & customers as an insurance agent. 

## Features
- Preprocesses text data from a CSV file.
- Encodes sender and receiver types for message threading.
- Trains a GPT-2 model with the Hugging Face Transformers library.
- Provides a user-friendly interface using Streamlit for text generation.
- Allows users to save the trained model and tokenizer.

---

## Installation

   Clone this repository:
   ```bash
   git clone https://github.com/sbuck4/streamlitsalesbot.git
   cd streamlitsalesbot
   ```

   Ensure you have TensorFlow and Streamlit installed:
   ```bash
   pip install tensorflow
   pip install streamlit
   ```

---

## File Structure

- **Resources/agency_mvp_text_threads.csv**: Contains the input text data for preprocessing and training.
- **app.py**: The main Streamlit application script.

---

## Preprocessing Steps

- Data is loaded from `Resources/agency_mvp_text_threads.csv`.
- Missing values in the `body` column are dropped.
- `sender_type` and `receiver_type` are encoded using `LabelEncoder`.
- A new column, `combined_text`, is created combining encoded sender, receiver, and message.

---

## Training the Model

1. The GPT-2 tokenizer and model are initialized:
   ```python
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   model = TFGPT2LMHeadModel.from_pretrained('gpt2')
   ```

2. The text is tokenized and converted into TensorFlow datasets.

3. The model is compiled using the Adam optimizer and trained for 3 epochs.

---

## Running the App

1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Enter a prompt in the text area and click "Generate" to see the generated text.

3. To train the model, click the "Train Model" button.

4. Save the trained model and tokenizer:
   ```python
   model.save_pretrained("./my_text_bot_model")
   tokenizer.save_pretrained("./my_text_bot_model")
   ```

---

## Dependencies

The following Python libraries are required:
- `pandas`
- `scikit-learn`
- `transformers`
- `tensorflow`
- `streamlit`

Install them using:
```bash
pip install pandas scikit-learn transformers tensorflow streamlit
```

---

## Example Code Snippets

### Loading and Preprocessing Data
```python
@st.cache_data
def load_data():
    text_df = pd.read_csv('Resources/agency_mvp_text_threads.csv')
    cleaned_text_df = text_df[['sender_type', 'receiver_type', 'body']].dropna(subset=['body'])
    ...
    return cleaned_text_df
```

### Training the Model
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss)
model.fit(train_dataset, validation_data=val_dataset, epochs=3)
```

### Generating Text
```python
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Contributing

Feel free to fork the repository and submit a pull request with improvements or bug fixes.

---

## License

This project is licensed under the MIT License.
