import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_metrics.json")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def data_split(sample_data):
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def train_model(data_split, sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    X_train, X_test, y_train, y_test = data_split

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def load_baseline_metrics():
    """ベースラインの性能指標を読み込み"""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def save_baseline_metrics(metrics):
    """ベースラインの性能指標を保存"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(metrics, f, indent=4)


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"

    # ベースラインの読み込み
    baseline_metrics = load_baseline_metrics()
    # jsonファイルが存在しない時
    if baseline_metrics is None:
        data = {"max_acc_model": {"accuracy": accuracy}}
        save_baseline_metrics(data)
        print(f"新しいベースラインを保存: accuracy = {accuracy}")
    else:
        if len(baseline_metrics) >= 1:
            pre_acc = baseline_metrics["max_acc_model"]["accuracy"]
        if "max_acc_model" not in baseline_metrics and len(baseline_metrics) == 1:
            pre_acc = baseline_metrics["default"]["accuracy"]
        # ベースラインと比較
        assert (
            accuracy >= pre_acc
        ), f"モデル精度が劣化しています:pre_acc={pre_acc},acc={accuracy}"
        baseline_metrics["max_acc_model"] = {"accuracy": accuracy}
        save_baseline_metrics(baseline_metrics)


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


@pytest.fixture
def compute_kl_divergence(data_split, bins=50, epsilon=1):
    X_train, X_test, y_train, y_test = data_split

    # 数値カラムだけ抽出し、PassengerIdを除外
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "PassengerId"]
    kl_divs = []

    for col in numeric_cols:
        train_feat = X_train[col].dropna()
        test_feat = X_test[col].dropna()

        # ヒストグラムで確率分布に変換（正規化）
        p_hist, _ = np.histogram(train_feat, bins=bins, density=True)
        q_hist, _ = np.histogram(test_feat, bins=bins, density=True)

        # ゼロ除算を避けるためにスムージング（epsilonを足す）
        p_hist += epsilon
        q_hist += epsilon

        # 正規化
        p = p_hist / np.sum(p_hist)
        q = q_hist / np.sum(q_hist)

        # KL divergence 計算
        kl = entropy(p, q)
        kl_divs.append((col, kl))

    return kl_divs


def test_kl_divergence(compute_kl_divergence, kl_threshold=0.5):
    """訓練データとテストデータ分布の確認"""
    kl_divs = compute_kl_divergence
    for col, kl in kl_divs:
        print(f"Feature {col} KL divergence: {kl:.4f}")
        assert kl < kl_threshold, f"Feature {col} KL divergence too high: {kl:.4f}"
