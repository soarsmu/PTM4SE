import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from Model.RoBERTa import RoBERTa
from utils.training_utils import load_checkpoint
from data.dataloader import get_train_valid_test,get_train_valid_Bert,get_train_valid_Bert_undersample
from torchsummary import summary
from Model.RoBERTa_v2 import RoBERTa_V2
import torch.nn.functional as F

def evaluate(model, test_loader, device):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for labels,data in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            data = data.type(torch.LongTensor)
            data = data.to(device)
            output = model(data, labels)

            _, output = output
            # print('out:', output)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
    print(y_true)
    print(y_pred)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0, 1], digits=4))

def evaluate_v2(model, test_loader, device):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for labels,data in test_loader:
            print(labels)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            data = data.type(torch.LongTensor)
            data = data.to(device)
            output = model(data)
            # print(output)
            # y_pred.extend(torch.argmax(output, 1).tolist())
            y_pred.extend(torch.max(output.data, 1)[1].cpu().numpy())
            y_true.extend(labels.tolist())
    print(y_true)
    print(y_pred)
    # for i in range(len(y_true))
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    datapath = os.path.join(os.getcwd(),'data')

    # bert_model = "roberta-base"

    bert_model = "bert-base-uncased"
    # bert_model = "bert-large-uncased"

    # bert = RoBERTa(bert_model=bert_model).to(device)
    # best_model = RoBERTa(bert_model=bert_model).to(device)
    best_model_V2 = RoBERTa_V2(bert_model=bert_model).to(device)
    
    
    train_iter, valid_iter, test_iter = get_train_valid_Bert_undersample(datapath, bert_model=bert_model, max_seq_lenght=128)

    train_iter, valid_iter, test_iter = get_train_valid_Bert(datapath, bert_model=bert_model, max_seq_lenght=128)
    # train_iter, valid_iter, test_iter = get_train_valid_test(datapath,bert_model=bert_model, max_seq_lenght=128)
    # load_checkpoint("data/models" + '/model.pt', best_model, device)
    load_checkpoint("data/models" + '/model.pt', best_model_V2, device)
    # evaluate(best_model, valid_iter, device)
    evaluate_v2(best_model_V2, test_iter, device)
