import torch
from sentence_transformers import SentenceTransformer
from torch import nn

# 确保这里的RUS_TO_LABEL与训练时使用的相同
RUS_TO_LABEL = {
    "Redundancy": 0,
    "Uniqueness": 1,
    "Synergy": 2
}

# 反转RUS_TO_LABEL字典，用于将数字标签转换回文本标签
LABEL_TO_RUS = {v: k for k, v in RUS_TO_LABEL.items()}

class Classifier(nn.Module):
    def __init__(self, transformer_model_name, num_classes):
        super(Classifier, self).__init__()
        self.transformer = SentenceTransformer(transformer_model_name)
        self.fc1 = nn.Linear(self.transformer.get_sentence_embedding_dimension(), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, sentences):
        embeddings = self.transformer.encode(sentences, convert_to_tensor=True)
        embeddings = embeddings.to(self.fc1.weight.device) 
        x = self.relu(self.fc1(embeddings))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

def load_model(model_path, device):
    num_classes = len(RUS_TO_LABEL)
    model = Classifier(transformer_model_name='sentence-transformers/all-distilroberta-v1', num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model, prompt, device):
    model.eval()
    with torch.no_grad():
        output = model([prompt])
        _, predicted = torch.max(output, 1)
        return LABEL_TO_RUS[predicted.item()]

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
model_path = 'runs/best_model.pt'  # 确保这是正确的模型保存路径
model = load_model(model_path, device)

# 测试函数
def test_prompt(prompt):
    prediction = predict(model, prompt, device)
    print(f"Prompt: {prompt}")
    print(f"Predicted category: {prediction}")
    print("---")

# 测试一些示例prompt
test_prompt("Given the Meme and the following caption\\nCaption: LOOK THERE MY FRIEND LIGHTYEAR NOW ALL SOHALIKUT TREND PLAY THE 10 YEARS CHALLENGE AT FACEBOOK imgflip.com.\\nQuestion1: How funny is the meme? Choose from the following comma separated options: funny, very_funny, not_funny, hilarious.")
test_prompt("Given an image and a question. Answer the question in a short answer. Question: How old do you have to be in canada to do this? Short Answer:")
test_prompt("The scientific species name of the species present in the image is:")
# # 交互式测试
# while True:
#     user_input = input("Enter a prompt (or 'quit' to exit): ")
#     if user_input.lower() == 'quit':
#         break
#     test_prompt(user_input)

print("Thank you for using the classifier!")