from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Function for model building and training
def train_and_evaluate(X_train, y_train_cat, X_val, y_val_cat, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\nModel Architecture:")
    model.summary()

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=30,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"\nFinal Validation Accuracy: {test_accuracy:.4f}")

    return model, history

model, history = train_and_evaluate(X_train, y_train_cat, X_test, y_test_cat, num_classes)
