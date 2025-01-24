import numpy as np
import pandas as pd
import logging
import config
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping


from datasets import load_data, preprocess_data, split_and_resample
from models import build_autoencoder_with_classifier
from utils import setup_logging, save_model_to_file, plot_precision_recall, plot_feature_sensitivity, \
    perform_feature_sensitivity_analysis


def main():
    """主函数，协调各个模块的工作"""
    # 1. 设置日志
    setup_logging()
    logging.info("项目开始运行")

    # 2. 加载和预处理数据
    logging.info("加载数据")
    data = load_data()

    logging.info("预处理数据")
    x, y, preprocessor = preprocess_data(data)

    # 3. 划分训练集和测试集，并进行SMOTE
    logging.info("划分训练集和测试集，并应用SMOTE")
    x_train, x_test, y_train, y_test = split_and_resample(x, y)

    logging.info(f"训练集大小: {x_train.shape}, 测试集大小: {x_test.shape}")
    logging.info(f"训练集类别分布:\n{pd.Series(y_train).value_counts()}")

    # 4. 构建模型
    logging.info("构建模型")
    input_dim = x_train.shape[1]
    model = build_autoencoder_with_classifier(
        input_dim=input_dim,
        encoding_dim=config.ENCODING_DIM,
        classification_loss_weight=config.CLASSIFICATION_LOSS_WEIGHT
    )

    model.summary(print_fn=logging.info)

    # 5. 训练模型
    logging.info("开始训练模型")
    history = model.fit(
        x_train,
        {
            "reconstruction": x_train,
            "classification": y_train
        },
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        verbose=1
    )

    # 6. 保存模型
    logging.info("保存训练好的模型")
    save_model_to_file(model, config.MODEL_SAVE_PATH)

    # 7. 评估模型
    logging.info("评估模型")
    reconstruction_pred, classification_pred = model.predict(x_test)
    threshold = 0.69
    y_pred = (classification_pred.ravel() > threshold).astype(int)

    logging.info(f"\nClassification Report (Threshold={threshold}):")
    report = classification_report(y_test, y_pred, target_names=["Normal", "Anomalous"])
    logging.info(report)

    logging.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"{cm}")

    # 8. 计算不同阈值下的Precision和Recall，并绘图
    logging.info("计算不同阈值下的Precision和Recall")
    thresholds = np.linspace(0.0, 1.0, 100)
    precision_list = []
    recall_list = []

    for thresh in thresholds:
        y_pred_thresh = (classification_pred.ravel() > thresh).astype(int)
        precision = precision_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)
        if abs(recall - precision) <= 0.05 and precision > 0:
            logging.info(f"we get a thresholds: {thresh}, while recall and precision = {recall}")

    plot_precision_recall(thresholds, precision_list, recall_list)

    # 9. 特征敏感度分析
    logging.info("开始特征敏感度分析")
    perform_feature_sensitivity_analysis(0.69)
    logging.info("项目运行结束")


if __name__ == "__main__":
    main()
