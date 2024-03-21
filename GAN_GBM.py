import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import anderson
import matplotlib.pyplot as plt
import seaborn as sns


class Phi(nn.Module):
    def __init__(self, input_size=3, n_hidden=2048):
        super(Phi, self).__init__()
        self.fc_input = nn.Linear(input_size, n_hidden)
        self.fc_hidden1 = nn.Linear(n_hidden, n_hidden)
        self.fc_hidden2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input_tensor):
        input_embedding = self.relu(self.fc_input(input_tensor))
        input_embedding = self.dropout(input_embedding)
        hidden1 = self.relu(self.fc_hidden1(input_embedding))
        hidden1 = self.dropout(hidden1)
        hidden2 = self.relu(self.fc_hidden2(hidden1))
        hidden2 = self.dropout(hidden2)
        output = self.fc_out(hidden2)
        return output

class F(nn.Module):
    def __init__(self, input_size=2, n_hidden=1024, activation='relu'):
        super(F, self).__init__()
        self.n_hidden = n_hidden
        self.fc_input = nn.Linear(input_size, n_hidden)
        self.fc_hidden1 = nn.Linear(n_hidden, n_hidden)
        self.fc_hidden2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, input_tensor):
        input_embedding = self.activation(self.fc_input(input_tensor))
        hidden1 = self.activation(self.fc_hidden1(input_embedding))
        hidden2 = self.activation(self.fc_hidden2(hidden1))
        output = self.fc_out(hidden2)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def CF_loss(X, Y, f):
    X, Y, f = X.view(X.size(0), -1), Y.view(Y.size(0), -1), f.view(f.size(0), -1)

    cos_X = (torch.cos(torch.matmul(f, X.t()))).mean(1)
    cos_Y = (torch.cos(torch.matmul(f, Y.t()))).mean(1)
    sin_X = (torch.sin(torch.matmul(f, X.t()))).mean(1)
    sin_Y = (torch.sin(torch.matmul(f, Y.t()))).mean(1)

    loss = (cos_X - cos_Y)**2 + (sin_X - sin_Y)**2

    return loss.mean()


def train_models(lr_g, lr_d, n_hidden, num_epochs, n_discriminator_iters):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_samples = 10000  # Number of samples taken

    u = torch.rand(num_samples, 1).to(device)

    sample = torch.rand(num_samples, 1)
    t = ((sample*100).floor() / 100).to(device)

    tp1 = t + 0.01

    sigma = (0.2 + 0.6 * torch.rand(num_samples, 1)).to(device)

    phi_input = torch.cat([u, t, sigma], dim=1).to(device)
    phi_p1_input = torch.cat([u, tp1, sigma], dim=1).to(device)

    phi_network = Phi().to(device)
    f_network = F(n_hidden=n_hidden).to(device)

    phi_optimizer = optim.Adam(phi_network.parameters(), lr=lr_g)
    f_optimizer = optim.Adam(f_network.parameters(), lr=lr_d)

    clip_value = 1
    print_every = 10

    for epoch in range(num_epochs):
        phi_output_t = phi_network(phi_input)
        phi_output_t_plus_1 = phi_network(phi_p1_input)
        dW_t = torch.randn_like(sigma).to(device)
        phi_output_t_with_noise = phi_output_t + sigma * dW_t


        for _ in range(n_discriminator_iters):
            f_optimizer.zero_grad()
            loss_discriminator = -CF_loss(phi_output_t_plus_1.detach(), phi_output_t_with_noise.detach(), f_network(torch.cat([t, sigma], dim=1)))
            loss_discriminator.backward()
            f_optimizer.step()

        additional_num_samples = 100
        add_t = torch.zeros(additional_num_samples, 1).to(device)
        add_sigma = torch.rand(additional_num_samples, 1).to(device)
        add_u = torch.rand(additional_num_samples, 1).to(device)
        add_phi_input = torch.cat([add_u, add_t, add_sigma], dim=1).to(device)


        phi_optimizer.zero_grad()
        loss_generator = CF_loss(phi_output_t_plus_1, phi_output_t_with_noise, f_network(torch.cat([t, sigma], dim=1)))
        add_loss_generator = nn.MSELoss()(phi_network(add_phi_input), torch.zeros_like(phi_network(add_phi_input))) ** 2
        loss_generator += add_loss_generator
        loss_generator.backward()
        phi_optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Discriminator Loss: {loss_discriminator.item()}, Generator Loss: {loss_generator.item()}')

    test_t = torch.full((num_samples, 1), 0.1).to(device)
    test_sigma = torch.full((num_samples, 1), 0.3).to(device)
    phi_input_test = torch.cat([u, test_t, test_sigma], dim=1)
    generated_samples_test = phi_network(phi_input_test).detach().cpu().numpy()

    ad_statistic_test, _, ad_critical_values_test = anderson(generated_samples_test.flatten(), dist='norm')
    mean = np.mean(generated_samples_test)
    variance = np.var(generated_samples_test)
    max_value = np.max(generated_samples_test)
    min_value = np.min(generated_samples_test)

    return mean, variance, max_value, min_value, ad_statistic_test, ad_critical_values_test


# lr_g_values = [0.0001, 0.00005, 0.00001]
# lr_d_values = [0.0001, 0.00005, 0.00001]
# n_hidden_values = [256, 512, 1024]
# num_epochs_values = [400, 700, 1000]
# n_discriminator_iters_values = [1,2,3]

lr_g_values = [0.000001]
lr_d_values = [0.00001]
n_hidden_values = [128,256]
num_epochs_values = [200, 400, 600]
n_discriminator_iters_values = [1, 2]

with open('results4_one.txt', 'w') as f:
    for lr_g in lr_g_values:
        for lr_d in lr_d_values:
            for n_hidden in n_hidden_values:
                for num_epochs in num_epochs_values:
                    for n_discriminator_iters in n_discriminator_iters_values:

                        mean, variance, max_value, min_value, ad_statistic_test, ad_critical_values_test = train_models(lr_g, lr_d, n_hidden, num_epochs, n_discriminator_iters)
                        f.write("#================================================================================#\n")
                        f.write("# lr_g: {} | lr_d: {} | n_hidden: {} \n".format(lr_g, lr_d, n_hidden))
                        f.write("# num_epochs: {} | n_disc_iters: {}   \n".format(num_epochs, n_discriminator_iters))
                        f.write("#-----------------------------------------------------#\n")
                        f.write("# Mean: {} | Variance: {} \n".format(mean, variance))
                        f.write("# Max: {} | Min: {}          \n".format(max_value, min_value))
                        f.write("# AD Statistic: {} | Critical Values: {} \n".format(ad_statistic_test, ad_critical_values_test))
                        f.write("#================================================================================#\n\n")

print("Training complete. Results saved to results.txt")