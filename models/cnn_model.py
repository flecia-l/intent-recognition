from .base_model import BaseIntentModel
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D, 
                                   Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils.glove_embeddings import load_glove_embeddings, create_embedding_matrix

class CNNIntentModel(BaseIntentModel):
    def __init__(self, vocab_size: int, num_classes: int, word_index: dict,
                 embedding_dim: int = 100, max_len: int = 30, glove_path: str = "data/glove.6B.100d.txt"):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.word_index = word_index
        self.glove_path = glove_path
        self.model = self.create_model()

    def create_model(self):
        embeddings_index = load_glove_embeddings(self.glove_path, self.embedding_dim)
        embedding_matrix = create_embedding_matrix(self.word_index, embeddings_index, self.embedding_dim)

        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix], trainable=False),
            Conv1D(128, 3, padding='valid', activation='relu'),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=0.0001)
        ]
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history.history

    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        return predicted_classes, confidences

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load_weights(path)