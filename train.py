import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

class QuantizedBertClassifier(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.quant = QuantStub()
        self.bert = bert_model
        self.dequant = DeQuantStub()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = self.quant(input_ids)
        if attention_mask is not None:
            attention_mask = self.quant(attention_mask)
            
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if labels is None:
            outputs.logits = self.dequant(outputs.logits)
        
        return outputs

class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def calibrate_model(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Calibrating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

def train_epoch(model, data_loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(predictions) == np.array(actual_labels))
    return total_loss / len(data_loader), accuracy

def train_sensitivity_classifier(train_texts, train_labels, val_texts=None, val_labels=None,
                               model_name='bert-base-uncased', epochs=3, batch_size=8,
                               learning_rate=2e-5, warmup_steps=0, quantization_aware=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    if quantization_aware:
        model = QuantizedBertClassifier(base_model)
        model.qconfig = get_default_qconfig('fbgemm' if device.type == 'cuda' else 'qnnpack')
        torch.quantization.prepare_qat(model, inplace=True)
    else:
        model = base_model
    
    model.to(device)
    
    if val_texts is None or val_labels is None:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42
        )
    
    train_dataset = CodeDataset(train_texts, train_labels, tokenizer)
    val_dataset = CodeDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    best_accuracy = 0
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Save the best model
            if quantization_aware:
                # Quantize the model before saving
                model.eval()
                torch.quantization.convert(model.to('cpu'), inplace=True)
                torch.save(model.state_dict(), 'best_model_quantized.pt')
                model.to(device)
                model.train()
            else:
                torch.save(model.state_dict(), 'best_model.pt')
    
    # Final quantization for deployment
    if quantization_aware:
        model.eval()
        model = model.to('cpu')
        torch.quantization.convert(model, inplace=True)
    
    return model, tokenizer

def predict_sensitivity(model, tokenizer, code_text, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    encoding = tokenizer.encode_plus(
        code_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1)
    
    return bool(prediction.item())

if __name__ == "__main__":
    code_samples = [
        "def get_user_password(): return 'password123'",
        "def calculate_sum(a, b): return a + b",
    ]
    
    labels = [1, 0]  # 1 for sensitive, 0 for non-sensitive
    
    model, tokenizer = train_sensitivity_classifier(
        train_texts=code_samples,
        train_labels=labels,
        epochs=3,
        batch_size=8,
        quantization_aware=True  # Enable quantization-aware training
    )
    
    test_code = "def authenticate(username, password): check_credentials(username, password)"
    is_sensitive = predict_sensitivity(model, tokenizer, test_code)
    print(f"Is code sensitive? {'Yes' if is_sensitive else 'No'}")
    
    torch.save(model.state_dict(), "quantized_model.pt")
    quantized_size = os.path.getsize("quantized_model.pt") / (1024 * 1024)  # Size in MB
    print(f"Quantized model size: {quantized_size:.2f} MB")