import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# Define a CNN for image feature extraction
class CNN(nn.Module):
    def __init__(self, embed_size):
        super(CNN, self).__init__()
        resnet = models.resnet152(pretrained=True)  # Load pre-trained ResNet-152
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# Define an RNN-based decoder
class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

# Load pre-trained vocabulary (you need to have this defined)
vocab_size = 10000  # Assuming vocabulary size
index2word = {}     # Dictionary to convert index to word

# Load the image and preprocess it
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Take input image path from the user
image_path = input("Enter the path to your input image: ")

# Load the image
try:
    image = Image.open(image_path)
except FileNotFoundError:
    print("File not found. Please provide a valid path to an image.")
    exit()

# Preprocess the image
image = transform(image).unsqueeze(0)  # Assuming 'image' is your input image
image_tensor = Variable(image).cuda() if torch.cuda.is_available() else Variable(image)

# Load pre-trained CNN
cnn = CNN(embed_size=256).cuda() if torch.cuda.is_available() else CNN(embed_size=256)
cnn.eval()

# Load pre-trained RNN
rnn = RNN(embed_size=256, hidden_size=512, vocab_size=vocab_size, num_layers=1).cuda() if torch.cuda.is_available() else RNN(embed_size=256, hidden_size=512, vocab_size=vocab_size, num_layers=1)
rnn.eval()

# Generate captions
features = cnn(image_tensor)
sampled_ids = []
inputs = torch.LongTensor([[1]]).cuda() if torch.cuda.is_available() else torch.LongTensor([[1]])
max_caption_length = 20  # Maximum length of the generated caption
for i in range(max_caption_length):
    outputs = rnn(features, inputs)
    _, predicted = outputs.max(2)
    sampled_ids.append(predicted.item())
    inputs = predicted  # Pass the current word as the next input

# Convert the sampled_ids to words
sampled_caption = []
for word_id in sampled_ids:
    word = index2word[word_id]
    sampled_caption.append(word)
    if word == '<end>':
        break
caption = ' '.join(sampled_caption[1:-1])  # Exclude <start> and <end>
print("Generated Caption:", caption)
