import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Chargement du dataset
df = pd.read_csv('./data/sentiment_data.csv')  # Modifier le nom du fichier si besoin
df = df[['Sentiment', 'Comment']].dropna()

le = LabelEncoder()
df['label'] = le.fit_transform(df['Sentiment'])  # negative: 0, neutral: 1, positive: 2

# Tokenizer simple - on doit construire le vocabulaire AVANT de créer les datasets
vocab = {'<pad>': 0}
def simple_tokenizer(text):
	idxs = []
	for w in text.lower().split():
		if w not in vocab:
			vocab[w] = len(vocab)
		idxs.append(vocab[w])
	return idxs

# Construire le vocabulaire sur tout le dataset
print("Construction du vocabulaire...")
for text in df['Comment']:
	simple_tokenizer(text)

print(f"Taille du vocabulaire: {len(vocab)}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Dataset PyTorch
class SentimentDataset(Dataset):
	def __init__(self, df, tokenizer, max_len=50):
		self.texts = df['Comment'].tolist()
		self.labels = df['label'].tolist()
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self): return len(self.labels)

	def __getitem__(self, idx):
		tokens = self.tokenizer(self.texts[idx])
		tokens = torch.tensor(tokens[:self.max_len], dtype=torch.long)
		label = torch.tensor(self.labels[idx], dtype=torch.long)
		return tokens, label

def collate_batch(batch):
	texts, labels = zip(*batch)
	texts = pad_sequence(texts, batch_first=True, padding_value=0)
	labels = torch.stack(labels)
	return texts, labels

train_ds = SentimentDataset(train_df, simple_tokenizer, max_len=50)
val_ds = SentimentDataset(val_df, simple_tokenizer, max_len=50)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

# 🚀 Modèle RNN + Self-Attention
class SelfAttention(nn.Module):
	def __init__(self, hidden_dim):
		super().__init__()
		self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
		self.scale = hidden_dim ** -0.5
	
	def forward(self, x):
		Q = self.query(x)
		K = self.key(x)
		V = self.value(x)
		scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
		w = F.softmax(scores, dim=-1)
		context = torch.matmul(w, V)
		return context

class SentimentClassifier(nn.Module):
	def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
		self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
		self.attn = SelfAttention(hidden_dim * 2)  # bidirectional donc hidden_dim * 2
		self.fc = nn.Linear(hidden_dim * 2, num_classes)
		self.dropout = nn.Dropout(0.3)
	
	def forward(self, x):
		# Embedding layer
		emb = self.embedding(x)  # [batch_size, seq_len, embed_dim]
		
		# RNN encoder
		rnn_out, _ = self.rnn(emb)  # [batch_size, seq_len, hidden_dim * 2]
		
		# Self-attention
		context = self.attn(rnn_out)  # [batch_size, seq_len, hidden_dim * 2]
		
		# Global average pooling
		pooled = context.mean(dim=1)  # [batch_size, hidden_dim * 2]
		
		# Dropout and classification
		pooled = self.dropout(pooled)
		logits = self.fc(pooled)  # [batch_size, num_classes]
		
		return logits

# Instanciation
VOCAB_SIZE = len(vocab)
model = SentimentClassifier(VOCAB_SIZE, embed_dim=100, hidden_dim=128, num_classes=len(le.classes_))

print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")

# Optimizer, Loss, Scheduler
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Entraînement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du device: {device}")
model.to(device)

for epoch in range(1, 4):
	model.train()
	total_loss = 0
	for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
		texts, labels = texts.to(device), labels.to(device)
		opt.zero_grad()
		logits = model(texts)
		loss = criterion(logits, labels)
		loss.backward()
		opt.step()
		total_loss += loss.item()
	avg_loss = total_loss / len(train_loader)
	print(f"Epoch {epoch} — Loss: {avg_loss:.4f}")

	# Évaluation
	model.eval()
	correct = total = 0
	with torch.no_grad():
		for texts, labels in val_loader:
			texts, labels = texts.to(device), labels.to(device)
			logits = model(texts)
			preds = logits.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	print(f"Validation Accuracy: {100*correct/total:.2f}%\n")
