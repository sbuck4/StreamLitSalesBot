{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting tf-keras\n",
      "  Downloading tf_keras-2.17.0-py3-none-any.whl.metadata (1.6 kB)\n",
      "Installing collected packages: tf-keras\n",
      "Successfully installed tf-keras-2.17.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "#pip install tf-keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from transformers import GPT2Tokenizer, TFGPT2LMHeadModel\n",
    "import tensorflow as tf\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-10-01 21:52:53.925 No runtime found, using MemoryCacheStorageManager\n",
      "2024-10-01 21:52:53.930 No runtime found, using MemoryCacheStorageManager\n"
     ]
    }
   ],
   "source": [
    "# Load and preprocess your data\n",
    "@st.cache_data\n",
    "def load_data():\n",
    "    text_df = pd.read_csv('Resources/agency_mvp_text_threads.csv')\n",
    "    cleaned_text_df = text_df[['sender_type', 'receiver_type', 'body']]\n",
    "    \n",
    "    cleaned_text_df = text_df[['sender_type', 'receiver_type', 'body']].copy()\n",
    "    cleaned_text_df = cleaned_text_df.dropna(subset=['body'])\n",
    "    \n",
    "    # Encode sender and receiver types\n",
    "    def safe_encode(series):\n",
    "        le = LabelEncoder()\n",
    "        series = series.fillna('Unknown')\n",
    "        le.fit(series)\n",
    "        return le.transform(series), le\n",
    "\n",
    "    sender_encoded, sender_encoder = safe_encode(cleaned_text_df['sender_type'])\n",
    "    receiver_encoded, receiver_encoder = safe_encode(cleaned_text_df['receiver_type'])\n",
    "\n",
    "    cleaned_text_df['sender_type_encoded'] = sender_encoded\n",
    "    cleaned_text_df['receiver_type_encoded'] = receiver_encoded\n",
    "\n",
    "    # Combine texts\n",
    "    cleaned_text_df['combined_text'] = cleaned_text_df.apply(\n",
    "        lambda row: f\"Sender: {row['sender_type_encoded']} Receiver: {row['receiver_type_encoded']} Message: {row['body']}\",\n",
    "        axis=1\n",
    "    )\n",
    "    return cleaned_text_df\n",
    "\n",
    "cleaned_text_df = load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the dataset into training and validation sets\n",
    "train_texts, val_texts = train_test_split(cleaned_text_df['combined_text'].tolist(), test_size=0.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\spenc\\anaconda3\\envs\\dev\\lib\\site-packages\\transformers\\tokenization_utils_base.py:1617: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be deprecated in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884\n",
      "  warnings.warn(\n",
      "All PyTorch model weights were used when initializing TFGPT2LMHeadModel.\n",
      "\n",
      "All the weights of TFGPT2LMHeadModel were initialized from the PyTorch model.\n",
      "If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2LMHeadModel for predictions without further training.\n"
     ]
    }
   ],
   "source": [
    "# Initialize the tokenizer and model\n",
    "tokenizer = GPT2Tokenizer.from_pretrained('gpt2')\n",
    "tokenizer.pad_token = tokenizer.eos_token  # Set pad token\n",
    "\n",
    "model = TFGPT2LMHeadModel.from_pretrained('gpt2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tokenize the inputs\n",
    "train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')\n",
    "val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='tf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert tokenized data into TensorFlow datasets\n",
    "train_dataset = tf.data.Dataset.from_tensor_slices((\n",
    "    dict(input_ids=train_encodings['input_ids'], attention_mask=train_encodings['attention_mask']),\n",
    "    train_encodings['input_ids']\n",
    "))\n",
    "val_dataset = tf.data.Dataset.from_tensor_slices((\n",
    "    dict(input_ids=val_encodings['input_ids'], attention_mask=val_encodings['attention_mask']),\n",
    "    val_encodings['input_ids']\n",
    "))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Batch the datasets and shuffle the training dataset\n",
    "train_dataset = train_dataset.shuffle(1000).batch(4)\n",
    "val_dataset = val_dataset.batch(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile the model with Adam optimizer and the built-in loss function\n",
    "optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)\n",
    "model.compile(optimizer=optimizer, loss=model.compute_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the model\n",
    "if st.button(\"Train Model\"):\n",
    "    model.fit(train_dataset,\n",
    "              validation_data=val_dataset,\n",
    "              epochs=3)\n",
    "    st.success(\"Model training complete!\")\n",
    "\n",
    "# Prediction function\n",
    "def generate_text(prompt):\n",
    "    input_ids = tokenizer.encode(prompt, return_tensors='tf')\n",
    "    output = model.generate(input_ids, max_length=50, num_return_sequences=1)\n",
    "    return tokenizer.decode(output[0], skip_special_tokens=True)\n",
    "\n",
    "# Streamlit user interface\n",
    "st.title(\"GPT-2 Model Text Generation\")\n",
    "st.write(\"Enter a prompt to generate text:\")\n",
    "\n",
    "user_input = st.text_area(\"Prompt:\", \"Type your prompt here...\")\n",
    "\n",
    "if st.button(\"Generate\"):\n",
    "    if user_input:\n",
    "        generated_text = generate_text(user_input)\n",
    "        st.write(\"Generated Text:\")\n",
    "        st.write(generated_text)\n",
    "    else:\n",
    "        st.warning(\"Please enter a prompt.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.save_pretrained(\"./my_text_bot_model\")\n",
    "tokenizer.save_pretrained(\"./my_text_bot_model\")\n",
    "st.success(\"Model and tokenizer saved!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
