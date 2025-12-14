# -*- coding: utf-8 -*-
"""chess_nn_large_scale.py

Updated with Rating Filters and Game Limits for faster processing.
"""

import os
import csv
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import chess
import chess.pgn
import chess.svg
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# ==========================================
# CONFIGURATION
# ==========================================
# UPDATE THIS to match your actual file name
PGN_FILE = "lichess.pgn"   
CSV_FILE = "training_data.csv"
MODEL_PATH = "chess_resnet_model.h5"

BATCH_SIZE = 2048            # Batch size for training
EPOCHS = 10                  # Training epochs

# NEW SETTINGS
MAX_GAMES_TO_PROCESS = 100000  # Stop after processing this many valid games
MIN_RATING = 2000              # Only use games where BOTH players are rated above this

# ==========================================
# 1. DATA PROCESSING (PGN -> CSV)
# ==========================================

def process_pgn_to_csv(pgn_path, csv_path):
    """
    Reads a local PGN file game-by-game and streams FEN/Result to a CSV.
    Filters by Rating and limits the total number of games.
    """
    print(f"Processing {pgn_path} into {csv_path}...")
    print(f"Filters: Min Rating > {MIN_RATING} | Limit: {MAX_GAMES_TO_PROCESS} games")
    
    if os.path.exists(csv_path):
        print(f"CSV {csv_path} already exists. Skipping processing.") 
        print("Delete the CSV file if you want to re-process the PGN.")
        return

    games_kept = 0
    games_scanned = 0
    position_count = 0
    
    # We use utf-8-sig to handle potentially weird encoding
    with open(pgn_path, encoding="utf-8-sig", errors="ignore") as pgn_file, open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FEN", "Result"]) # CSV Header

        while games_kept < MAX_GAMES_TO_PROCESS:
            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception:
                continue # Skip corrupt PGN lines/games
            
            if game is None:
                break # End of file

            games_scanned += 1

            # 1. FILTER: CHECK RATINGS
            # Lichess PGNs use "WhiteElo" and "BlackElo" headers
            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
                
                if white_elo < MIN_RATING or black_elo < MIN_RATING:
                    continue # Skip low rated games
            except ValueError:
                continue # Skip games with weird/missing ratings

            # 2. FILTER: CHECK RESULT
            result = game.headers.get("Result", "*")
            if result == "1-0":
                label = 1.0
            elif result == "0-1":
                label = 0.0
            elif result == "1/2-1/2":
                label = 0.5
            else:
                continue # Skip games with unknown results

            # 3. REPLAY GAME (Only if filters passed)
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                
                # FILTER: Skip early opening moves (first 5 full moves)
                if board.fullmove_number < 6:
                    continue
                
                writer.writerow([board.fen(), label])
                position_count += 1

            games_kept += 1
            
            # Status Update every 1000 games kept
            if games_kept % 1000 == 0:
                print(f"Kept {games_kept} / {games_scanned} scanned | Positions: {position_count}")

    print(f"Done. Processed {games_kept} high-quality games. Extracted {position_count} positions.")

# ==========================================
# 2. HELPER: BOARD REPRESENTATION
# ==========================================

def board_to_matrix(board):
    """Converts a chess.Board object to an 8x8x12 numpy matrix."""
    matrix = np.zeros((8, 8, 12), dtype=np.int8)
    piece_type_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # 0-5 for White, 6-11 for Black
            layer = piece_type_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            matrix[row, col, layer] = 1
    return matrix

# ==========================================
# 3. DATA GENERATOR (Streaming Training)
# ==========================================

class ChessDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Generator to load data in batches from the CSV.
    Prevents OOM (Out of Memory) errors with large datasets.
    """
    def __init__(self, csv_file, batch_size=2048, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        print("Loading CSV indices into memory...")
        # Pandas reads the CSV columns very efficiently
        self.df = pd.read_csv(csv_file)
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
        print(f"Generator ready with {len(self.df)} samples.")

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # Pick indices for this batch
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        
        # Get data
        batch_fens = self.df.iloc[batch_indices]['FEN'].values
        batch_labels = self.df.iloc[batch_indices]['Result'].values.astype('float32')
        
        # Convert FENs to Matrices on the fly
        X_batch = np.array([self._fen_to_mat(f) for f in batch_fens])
        
        return X_batch, batch_labels

    def _fen_to_mat(self, fen):
        return board_to_matrix(chess.Board(fen))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==========================================
# 4. MODEL DEFINITION
# ==========================================

def build_resnet_model():
    """Builds the ResNet Chess Model."""
    def residual_block(x, filters):
        shortcut = x
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=(8, 8, 12))
    
    # Initial Conv
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 4 Residual Blocks
    for _ in range(4):
        x = residual_block(x, 64)

    # Head
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Chess_ResNet")
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# ==========================================
# 5. EVALUATION TOOLS
# ==========================================

def evaluate_openings(model):
    """Evaluates common openings to see model bias."""
    opening_moves = {
        'Ruy Lopez': 'e4 e5 Nf3 Nc6 Bb5',
        'Sicilian Defense': 'e4 c5',
        "Queen's Gambit": 'd4 d5 c4',
        'French Defense': 'e4 e6',
        'Caro-Kann': 'e4 c6',
        'Italian Game': 'e4 e5 Nf3 Nc6 Bc4',
        "King's Indian": 'd4 Nf6 c4 g6',
        'English Opening': 'c4'
    }
    
    print("\n--- Opening Evaluation ---")
    results = []
    for name, moves in opening_moves.items():
        board = chess.Board()
        for move in moves.split():
            board.push_san(move)
        
        matrix = board_to_matrix(board)
        # Add batch dim
        pred = model.predict(np.expand_dims(matrix, axis=0), verbose=0)[0][0]
        results.append((name, pred))
        
    # Sort and Print
    results.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(results, columns=["Opening", "White Win Prob"])
    print(df)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.barh(df["Opening"], df["White Win Prob"], color='skyblue')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title("Model Evaluation of Openings")
    plt.gca().invert_yaxis()
    plt.show()

def generate_saliency_map(fen, model):
    """Generates saliency map for a FEN."""
    board = chess.Board(fen)
    matrix = board_to_matrix(board)
    input_tensor = tf.convert_to_tensor(np.expand_dims(matrix, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)

    gradients = tape.gradient(prediction, input_tensor)
    saliency = np.sum(np.abs(gradients[0]), axis=-1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(saliency, cmap='hot', interpolation='nearest')
    plt.title(f"Saliency Map")
    plt.colorbar()
    plt.show()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():
    # 1. Check for File
    if not os.path.exists(PGN_FILE):
        print(f"CRITICAL ERROR: Could not find '{PGN_FILE}'")
        print("Please rename your downloaded file to 'large_dataset.pgn' or update the PGN_FILE variable.")
        return

    # 2. Process Data
    print("\nStep 1: Processing PGN to CSV (Importing)...")
    process_pgn_to_csv(PGN_FILE, CSV_FILE)
    
    # 3. Initialize Generators
    print("\nStep 2: Initializing Data Generator...")
    if not os.path.exists(CSV_FILE):
        print("Error: CSV file not found. Processing failed.")
        return

    train_gen = ChessDataGenerator(CSV_FILE, batch_size=BATCH_SIZE)
    
    # 4. Build/Load Model
    print("\nStep 3: Building Model...")
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
    else:
        print("Creating new model...")
        model = build_resnet_model()
        model.summary()

        # 5. Train
        print("\nStep 4: Training...")
        try:
            history = model.fit(
                train_gen,
                epochs=EPOCHS,
                verbose=1
            )
            model.save(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
            
            # Plot History
            plt.plot(history.history['loss'], label='Loss')
            plt.plot(history.history['mae'], label='MAE')
            plt.legend()
            plt.show()
            
        except KeyboardInterrupt:
            print("Training interrupted. Saving current state...")
            model.save(MODEL_PATH)

    # 6. Evaluate
    print("\nStep 5: Evaluation...")
    evaluate_openings(model)
    
    # Saliency Check (Start Position)
    generate_saliency_map(chess.STARTING_FEN, model)

if __name__ == "__main__":
    main()