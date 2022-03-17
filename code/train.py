from torch_geometric.data import DataLoader
from utils import *
from models import *
import argparse
import time, datetime
from sklearn.metrics import f1_score
import os
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='EModel', help='model name')
parser.add_argument('--dataset', type=str, default="PSD", help='dataset name')
parser.add_argument('--nhid', type=int, default=512, help='hidden size')
parser.add_argument('--num_features', type=int, default=20, help='num_feature')
parser.add_argument('--num_classes', type=int, default=54, help='num_classes')
parser.add_argument('--num_nodes', type=int, default=76, help='num_nodes,121 for 719, 24 for cva')
parser.add_argument('--batch_size', type=int, default=60, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs')
parser.add_argument('--num_heads', type=int, default=3, help='maximum number of epochs')


args = parser.parse_args()
args.use_gpu = torch.cuda.is_available()
print(args)

gpu_ids = [0]
device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else "cpu")

now = int(time.time())
timeArray = time.localtime(now)
Time = time.strftime("%Y%m%d%H%M", timeArray)

ex_path = './log/' + args.dataset + args.model_name + '_' + Time + '/'
if os.path.exists(ex_path) == False:
    os.makedirs(ex_path)

log_file = ex_path + args.dataset + Time + '_log' + '.txt'
print(log_file)

log = open(log_file, 'w')
print(args, file=log)
print(log_file, file=log)

model_name_acc = ex_path + args.dataset + Time + args.model_name + '_acc.pkl'
print(model_name_acc, file=log)

train_set = data_process(raw_path='./data/train_set_psd.mat', num_nodes=args.num_nodes)
test_set = data_process(raw_path='./data/test_set_psd.mat', num_nodes=args.num_nodes)
      
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

num_train = len(train_set)
num_test = len(test_set)
print('num_train', num_train)
print('num_test', num_test)

model_file = __import__('models')
model_cls_name = args.model_name
NET = getattr(model_file, model_cls_name)
model = NET(args)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch, best_acc):
    model.train()
    train_loss = 0
    train_acc = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        if args.use_gpu:
            data = data.to(device)
        prediction = model(data)
        loss = F.nll_loss(prediction, data.y)
        _, pred = prediction.max(dim=1)
        acc = (pred == data.y).sum()
        train_loss += loss.detach().item()
        train_acc += acc.item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / num_train
    train_acc = train_acc / num_train

    test_loss = 0
    test_acc = 0
    preddd = torch.tensor([0])
    yy = torch.tensor([0])

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if args.use_gpu:
                data = data.to(device)
            prediction = model(data)
            loss = F.nll_loss(prediction, data.y)
            _, pred = prediction.max(dim=1)
            acc = (pred == data.y).sum()
            preddd = torch.cat([preddd, pred.cpu()])
            yy = torch.cat([yy, data.y.cpu()])
            test_loss += loss.detach().item()
            test_acc += acc.item()

    test_loss = test_loss / num_test
    test_acc = test_acc / num_test
    
    if test_acc > best_acc:
        torch.save(model.state_dict(), model_name_acc)
        print('best valid epoch so far, saving...', epoch + 1)
    
    preddd = preddd[1::]
    yy = yy[1::]
    micro_f1 = f1_score(preddd, yy, average='micro')
    macro_f1 = f1_score(preddd, yy, average='macro')

    print('epoch{}, train_Loss: {:.6f}, train_Acc: {:.6f}, '
          'test_loss: {:.6f}, test_Acc: {:.6f}, micro_f1: {:.6f}, macro_f1: {:.6f}'.format(
        epoch + 1, train_loss, train_acc, test_loss, test_acc, micro_f1, macro_f1))
    print('epoch{}, train_Loss: {:.6f}, train_Acc: {:.6f}, '
          'test_loss: {:.6f}, test_Acc: {:.6f}, micro_f1: {:.6f}, macro_f1: {:.6f}'.format(
        epoch + 1, train_loss, train_acc, test_loss, test_acc, micro_f1, macro_f1), file=log)
    log.flush()

    return test_acc


print('start training')
best_acc = 0
for epoch in range(args.epochs):
    test_acc = train(epoch, best_acc)
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch_acc = epoch


now = int(time.time())
timeArray = time.localtime(now)
print(Time)
print(time.strftime("%Y%m%d_%H%M", timeArray))
print(time.strftime("%Y%m%d_%H%M", timeArray), file=log)
log.flush()










