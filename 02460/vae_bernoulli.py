# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.randn(self.M), requires_grad=False)
        self.logstd = nn.Parameter(torch.randn(self.M), requires_grad=False)
        self.name = 'GaussianPrior'

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=torch.exp(self.logstd)), 1)

class MixturesOfGaussiansPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a mixture of Gaussians prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int] 
           Number of mixture components.
        """
        super(MixturesOfGaussiansPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.randn(self.K,self.M))
        self.logstd = nn.Parameter(torch.randn(self.K,self.M))
        self.pi = nn.Parameter(torch.zeros(self.K,)/self.K)
        self.name = 'MixturesOfGaussiansPrior'

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(F.softmax(self.pi,dim=0))
        comp = td.Independent(td.Normal(loc=self.mean, scale=torch.exp(self.logstd)), 1)
        return td.MixtureSameFamily(mix, comp, 1)
        # return td.MixtureSameFamily(td.Categorical(self.pi), td.Independent(td.Normal(loc=self.mean, scale=self.std), 1), 1)
        # return td.MixtureSameFamily(td.Categorical(self.pi), td.Independent(td.Normal(loc=self.mean, scale=self.std)), 1)
        # return td.independent(td.mixture_same_family.MixtureSameFamily(td.Categorical(torch.ones(self.K)), td.Normal(loc=self.mean, scale=self.std)), 1)
        # return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x, n_samples = 10):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        if isinstance(self.prior, MixturesOfGaussiansPrior):
            # approximate KL divergence by sampling p(x) and evaluating log(p(x)/q(x))
            # p_log_probs = torch.zeros(n_samples,x.shape[0])
            # q_log_probs = torch.zeros(n_samples,x.shape[0])
            # for i in range(n_samples):
            #     samples = self.prior().sample((x.shape[0],))
            #     p_log_probs[i] = self.prior.forward().log_prob(samples) 
            #     q_log_probs[i] = q.log_prob(samples)
            # KLdiv = p_log_probs.mean(dim=0) - q_log_probs.mean(dim=0)

            log_pxz = self.decoder(z).log_prob(x)
            log_qzx = q.log_prob(z)
            log_pz = self.prior().log_prob(z)
            elbo = torch.mean(log_pxz - log_qzx + log_pz)

        else:
            KLdiv = td.kl_divergence(q, self.prior())
            elbo = torch.mean(self.decoder(z).log_prob(x) - KLdiv, dim=0)
        
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader)*epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='plot_approx_posterior', choices=['train', 'sample','plot_approx_posterior','eval_test_set'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='mog', choices=['gaussian', 'mog'], help='prior distribution (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
        savename = 'gaussian'
    elif args.prior == 'mog':
        prior = MixturesOfGaussiansPrior(M, 3)
        savename = 'mog'

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), savename+'_model.pt')

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(savename+'_model.pt', map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), savename+'_samples.png')
    elif args.mode == 'eval_test_set':
        model.load_state_dict(torch.load(savename+'_model.pt', map_location=torch.device(args.device)))

        model.eval()
        # Evaluate test set
        with torch.no_grad():
            test_loss = 0
            for step,(x, y) in enumerate(mnist_test_loader):
                x = x.to(device)
                test_loss += model.elbo(x).item()
            test_loss /= len(mnist_test_loader.dataset)
            print(f"Test loss: {test_loss:.3f}")
    elif args.mode == 'plot_approx_posterior':
        import seaborn as sns
        model.load_state_dict(torch.load(savename+'_model.pt', map_location=torch.device(args.device)))

        model.eval()
        # Plot approx posterior for the test set
        with torch.no_grad():
            all_z = torch.zeros((10000,M))
            for step,(x, y) in enumerate(mnist_test_loader):
                x = x.to(device)
                q = model.encoder(x)
                z = q.rsample()
                all_z[step*args.batch_size:(step+1)*args.batch_size,:] = z
            
            if M>2:
                #do PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                z = pca.fit_transform(all_z)
            else:
                z = all_z
            
            n_samples = 10000
            p = model.prior().sample(torch.Size([n_samples]))
            sns.kdeplot(x=p[:,0],y=p[:,1])

            y = torch.cat([y for x,y in mnist_test_loader])
            sns.kdeplot
            plt.scatter(z[:, 0], z[:, 1], c=y,marker='.',cmap='tab10',alpha=0.5, s=1)
            plt.colorbar()
            plt.savefig(savename+'_approx_posterior.png')
            plt.close()
        # decoder(z).log_prob(x)
