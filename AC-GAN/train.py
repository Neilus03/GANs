import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model import Generator, Discriminator

# Hyperparameters
batch_size = 16
lr = 0.0002
noise_dim = 100
class_dim = 3  # Number of classes
image_shape = 560 * 640 * 3  # Flatten the images
num_epochs = 200

# Initialize models
generator = Generator(noise_dim, class_dim, image_shape)
discriminator = Discriminator(image_shape, class_dim)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Load your data here as train_loader
# train_loader = ...

# Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(torch.FloatTensor))
        labels = Variable(labels.type(torch.LongTensor))

        # Train Generator
        optimizer_G.zero_grad()

        noise = Variable(torch.FloatTensor(torch.randn(batch_size, noise_dim)))
        gen_labels = Variable(torch.LongTensor(torch.randint(0, class_dim, (batch_size,))))
        gen_labels_onehot = F.one_hot(gen_labels, num_classes=class_dim).float()

        gen_imgs = generator(noise, gen_labels_onehot)
        validity, pred_label = discriminator(gen_imgs)

        g_loss = adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)

        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

