# models.py

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import config

def build_autoencoder_with_classifier(input_dim, encoding_dim=16, classification_loss_weight=1.0):
    """
    构建联合模型，自编码器与分类器共享编码层：
      - reconstruction: MSE
      - classification: BCE
    loss_weights 中可增大 'classification' 的权重
    """
    input_layer = layers.Input(shape=(input_dim,), name="input_layer")

    # 编码器
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu', name="encoded")(encoded)

    # 解码器(重构)
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid', name="reconstruction")(decoded)

    # 分类器
    classification = layers.Dense(64, activation='relu')(encoded)
    classification = layers.Dense(1, activation='sigmoid', name="classification")(classification)

    # 模型
    model = models.Model(inputs=input_layer, outputs=[decoded, classification])

    # 设置两个分支的损失及其权重
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss={
            "reconstruction": "mse",
            "classification": "binary_crossentropy"
        },
        loss_weights={
            "reconstruction": 0.5,
            "classification": classification_loss_weight
        },
        metrics={
            "classification": "accuracy"
        }
    )
    return model
