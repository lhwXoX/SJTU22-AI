import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from colored_mnist import ColoredMNIST
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class ConvNet_SNN(nn.Module):
  def __init__(self, T: int, channels = 32):
    super().__init__()
    self.T = T
    self.conv_fc = nn.Sequential(
    layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
    layer.BatchNorm2d(channels),
    neuron.IFNode(surrogate_function=surrogate.ATan()),
    layer.MaxPool2d(2, 2),  # 14 * 14
    layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
    layer.BatchNorm2d(channels),
    neuron.IFNode(surrogate_function=surrogate.ATan()),
    layer.MaxPool2d(2, 2),  # 7 * 7
    layer.Flatten(),
    )
    self.conv2 = nn.Sequential(
    layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
    neuron.IFNode(surrogate_function=surrogate.ATan()),
    )
    self.last_linear = nn.Sequential(
    layer.Linear(channels * 4 * 4, 1, bias=False),
    )
    functional.set_step_mode(self, step_mode='m')
  
  def forward(self, x: torch.Tensor):
    # x.shape = [N, C, H, W]
    x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
    x_seq = self.conv_fc(x_seq) #[T, N, 1]
    x_seq2 = self.conv2(x_seq)
    out = self.last_linear(x_seq2)
    #print(x_seq.mean(0))
    out = out.mean(0).flatten()
    return x_seq, x_seq2, out

def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()

def test_model(model, device, test_loader, set_name="test set"):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      _, _, output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)


def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()

def compute_snn_penalty(seq_1, seq_2):
  x_seq1 = seq_1.squeeze(-1)  # [T, N]
  x_seq2 = seq_2.squeeze(-1)  # [T, N]
  cosine_similarities = torch.nn.functional.cosine_similarity(x_seq1, x_seq2, dim=2)  # [T]
  mapped_sim = cosine_similarities
  loss = (1 - mapped_sim).mean()
  return loss

def irm_train(model, device, train_loaders, test_loader, optimizer, epoch):
  model.train()
  train_loaders = [iter(x) for x in train_loaders]
  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)
  batch_idx = 0
  #penalty_multiplier = (epoch-1) ** 1.6
  penalty_multiplier = (epoch - 1) / 5 #lambda linear
  print(f'Using penalty multiplier {penalty_multiplier}')
  while True:
    optimizer.zero_grad()
    error = 0
    penalty = 0
    seq = []
    seq1 = []
    for loader in train_loaders:
      data, target = next(loader, (None, None))
      #print(target)
      if data is None:
        return
      data, target = data.to(device), target.to(device).float()
      res1, res, output = model(data)
      seq1.append(res1)
      seq.append(res)
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
      penalty += compute_irm_penalty(loss_erm, dummy_w)
      error += loss_erm.mean()
    snn_penalty1 = compute_snn_penalty(seq1[0], seq1[1])
    snn_penalty2 = compute_snn_penalty(seq[0], seq[1])
    snn_penalty = (snn_penalty1 * 0.3 + snn_penalty2 * 0.7)
    #print(error, penalty, snn_penalty)
    
    if penalty_multiplier > 1:
      ((error + snn_penalty * penalty_multiplier) /penalty_multiplier).backward()
    else:
      (error + snn_penalty * penalty_multiplier).backward() 
    
    #(error + penalty_multiplier * snn_penalty).backward()
    optimizer.step()
    functional.reset_net(model)
    
    if batch_idx % 79 == 0 and batch_idx != 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tSNN penalty: {:.6f}\tpenalty: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loaders[0]) * len(data),
               100. * batch_idx / len(train_loaders[0]), error.item(), snn_penalty.item(), penalty.item()))
      #print('First 20 logits', output.data.cpu().numpy()[:20])
    

    batch_idx += 1
    return snn_penalty.item()


def plot_line(plot_x, train_acc, test_acc, out_dir):
    plt.plot(plot_x, train_acc["train1"], label="Train1")
    plt.plot(plot_x, train_acc["train2"], label="Train2")
    plt.plot(plot_x, test_acc, label="Test")
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    location = 'upper left'
    plt.legend(loc=location)
    plt.title("Acc Curve")
    plt.savefig(out_dir)
    plt.close()
    
import matplotlib.pyplot as plt

def plot_snn(plot_x, test_acc, snn, out_dir):
    fig, ax1 = plt.subplots()
    ax1.plot(plot_x, snn, label="Similarity Loss", color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(plot_x, test_acc, label="Test Accuracy", color='tab:orange')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title("Accuracy and Loss Curve")
    plt.savefig(out_dir)
    plt.close()


def train_and_test_irm():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train1_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     #transforms.Resize(32),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=250, shuffle=True, **kwargs)

  train2_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train2',
                 transform=transforms.Compose([
                     #transforms.Resize(32),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=250, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      #transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=250, shuffle=True, **kwargs)

  model = ConvNet_SNN(T=8).to(device)
  #model = resnet20().to(device)
  #model = VGG("VGG11", T=4).to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  train_acc = {"train1": [], "train2": []}
  test_set_acc = []
  snn = []
  for epoch in range(1, 51):
    snn_penalty = irm_train(model, device, [train1_loader, train2_loader], test_loader, optimizer, epoch)
    train1_acc = test_model(model, device, train1_loader, set_name='train1 set')
    train2_acc = test_model(model, device, train2_loader, set_name='train2 set')
    test_acc = test_model(model, device, test_loader)
    train_acc["train1"].append(train1_acc), train_acc["train2"].append(train2_acc)
    snn.append(snn_penalty)
    test_set_acc.append(test_acc)
    print(f"epoch: {epoch}, train1 acc: {train1_acc}, train2_acc: {train2_acc}, test_acc: {test_acc}")

    #print('found acceptable values. stopping training.')
    plot_x = range(len(test_set_acc))
      #画曲线
    plot_line(plot_x, train_acc, test_set_acc, out_dir='E:/Homework/ZN/IRM/output/acc.png')
    plot_snn(plot_x, test_set_acc, snn, out_dir='E:/Homework/ZN/IRM/output/acc_snn.png')
    if test_acc > 60:
      return

def main():
  train_and_test_irm()
  #train_and_test_erm()

if __name__ == '__main__':
  main()