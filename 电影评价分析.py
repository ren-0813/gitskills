import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import fasttext
import matplotlib.pyplot as plt
import re
from collections import Counter
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SimpleDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab = vocab or self._build_vocab(texts)
        self.vocab_size = len(self.vocab)

    def _build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            words = re.sub(r'[^\w\s]', '', text.lower()).split()
            word_counts.update(words)

        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common(10000):
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _text_to_sequence(self, text):
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        sequence = [self.vocab.get(word, 1) for word in words]
        if len(sequence) < self.max_length:
            sequence += [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        return sequence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sequence = self._text_to_sequence(self.texts[idx])
        return torch.tensor(sequence), torch.tensor(self.labels[idx], dtype=torch.long)


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2,
                 n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))


class FastTextModel:
    def __init__(self):
        self.model = None
        self.losses = []
        self.accuracies = []

    def _prepare_data(self, texts, labels, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for text, label in zip(texts, labels):
                label_str = '__label__positive' if label == 1 else '__label__negative'
                text_clean = re.sub(r'\s+', ' ', text.strip())
                f.write(f"{label_str} {text_clean}\n")

    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=10, lr=0.1):
        self.losses, self.accuracies = [], []

        # 准备数据文件
        train_file, val_file = 'fasttext_train.txt', 'fasttext_val.txt'
        self._prepare_data(train_texts, train_labels, train_file)
        self._prepare_data(val_texts, val_labels, val_file)

        for epoch in range(1, epochs + 1):
            self.model = fasttext.train_supervised(
                input=train_file, epoch=1, lr=lr, wordNgrams=2, verbose=0
            )

            # 计算验证指标
            try:
                result = self.model.test(val_file)
                acc = result[1]
                loss = 1 - acc
            except:
                loss, acc = 1.0, 0.0

            self.losses.append(loss)
            self.accuracies.append(acc)
            print(f"FastText Epoch {epoch}: Loss: {loss:.4f}, Acc: {acc:.4f}")

        # 清理临时文件
        for f in [train_file, val_file]:
            if os.path.exists(f):
                os.remove(f)

        return self.losses, self.accuracies

    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained!")

        preds = []
        for text in texts:
            try:
                label = self.model.predict(text.strip())[0][0]
                preds.append(1 if '__label__positive' in label else 0)
            except:
                preds.append(0)
        return preds


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds), all_preds, all_labels


def train_rnn(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for texts, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, _, _ = evaluate(model, val_loader, device)

        losses.append(avg_loss)
        accuracies.append(val_acc)
        print(f'Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')

    return losses, accuracies


def plot_results(rnn_losses, rnn_accs, fasttext_losses, fasttext_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失对比
    epochs = range(1, len(rnn_losses) + 1)
    ax1.plot(epochs, rnn_losses, 'b-', label='TextRNN Loss', marker='o')
    if fasttext_losses:
        min_len = min(len(rnn_losses), len(fasttext_losses))
        ax1.plot(epochs[:min_len], fasttext_losses[:min_len], 'r-', label='FastText Loss', marker='s')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率对比
    ax2.plot(epochs, rnn_accs, 'b-', label='TextRNN Acc', marker='o')
    if fasttext_accs:
        min_len = min(len(rnn_accs), len(fasttext_accs))
        ax2.plot(epochs[:min_len], fasttext_accs[:min_len], 'r-', label='FastText Acc', marker='s')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def load_data(file_path):
    try:
        # 尝试不同的分隔符
        for sep in ['\t', ',', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None, names=['text', 'sentiment'])
                break
            except:
                continue
        else:
            raise ValueError("无法读取文件")
    except:
        print("使用示例数据")
        return create_sample_data()

    texts = df['text'].astype(str).tolist()
    labels = []

    for sentiment in df['sentiment']:
        try:
            # 数值标签
            sentiment_val = float(sentiment)
            labels.append(0 if sentiment_val <= 2 else 1)
        except:
            # 文本标签
            sentiment_str = str(sentiment).lower()
            labels.append(0 if sentiment_str in ['negative', 'neg', '0', 'false', 'bad'] else 1)

    return texts, labels


def create_sample_data():
    positive = ["Great movie!", "Excellent acting", "Wonderful story"]
    negative = ["Terrible film", "Boring plot", "Bad acting"]
    texts = positive + negative
    labels = [1] * 3 + [0] * 3
    return texts, labels


def main():
    # 设置
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    try:
        texts, labels = load_data('C:\\Users\\hm943\\Downloads\\train.tsv\\train.txt')
    except:
        texts, labels = create_sample_data()

    print(f"数据量: {len(texts)}, 正面: {sum(labels)}, 负面: {len(labels) - sum(labels)}")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2,
                                                        random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=42, stratify=y_train)

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 训练FastText
    print("\n训练FastText...")
    fasttext_model = FastTextModel()
    fasttext_losses, fasttext_accs = fasttext_model.train(X_train, y_train, X_val, y_val, epochs=10)
    fasttext_val_preds = fasttext_model.predict(X_val)
    fasttext_val_acc = accuracy_score(y_val, fasttext_val_preds)
    print(f"FastText验证准确率: {fasttext_val_acc:.4f}")

    # 训练TextRNN
    print("\n训练TextRNN...")
    train_dataset = SimpleDataset(X_train, y_train, max_length=50)
    val_dataset = SimpleDataset(X_val, y_val, vocab=train_dataset.vocab, max_length=50)
    test_dataset = SimpleDataset(X_test, y_test, vocab=train_dataset.vocab, max_length=50)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    rnn_model = SimpleRNN(vocab_size=train_dataset.vocab_size).to(device)
    rnn_losses, rnn_accs = train_rnn(rnn_model, train_loader, val_loader, device, epochs=10)

    test_acc, test_preds, test_labels = evaluate(rnn_model, test_loader, device)
    print(f"TextRNN测试准确率: {test_acc:.4f}")

    # 绘制结果
    plot_results(rnn_losses, rnn_accs, fasttext_losses, fasttext_accs)

    # 结果对比
    print("\n模型对比:")
    print(f"FastText验证准确率: {fasttext_val_acc:.4f}")
    print(f"TextRNN最佳验证准确率: {max(rnn_accs):.4f}")
    print(f"TextRNN测试准确率: {test_acc:.4f}")

    # 保存模型
    torch.save(rnn_model.state_dict(), 'textrnn_model.pth')
    print("模型已保存")

    # 预测示例
    print("\n预测示例:")
    sample_idx = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)
    for i, idx in enumerate(sample_idx):
        text, true_label = X_test[idx], y_test[idx]

        fasttext_pred = fasttext_model.predict([text])[0]

        rnn_model.eval()
        with torch.no_grad():
            sequence = test_dataset._text_to_sequence(text)
            sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
            rnn_pred = torch.argmax(rnn_model(sequence_tensor), 1).item()

        print(f"\n示例 {i + 1}:")
        print(f"文本: {text[:80]}...")
        print(f"真实: {'正面' if true_label == 1 else '负面'}")
        print(f"FastText: {'正面' if fasttext_pred == 1 else '负面'}")
        print(f"TextRNN: {'正面' if rnn_pred == 1 else '负面'}")


if __name__ == "__main__":
    main()