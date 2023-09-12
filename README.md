# Simple GAN Implementation using PyTorch

## Overview

This repository contains a simple Generative Adversarial Network (GAN) implemented in PyTorch. The GAN is trained on the MNIST dataset to generate handwritten digits.

## Features

- Built using PyTorch.
- Leverages the `nn.Module` class for building the Generator and Discriminator networks.
- Uses Binary Cross Entropy loss and the Adam optimizer.
- Hyperparameters are easily configurable.
- Logs GAN training in real-time using TensorBoard.
- Notes on potential improvements in the script comments.

## Installation

You will need Python 3.x and PyTorch installed to run this code. You can install the required packages using pip:

\```bash
pip install torch torchvision
pip install tensorboard
\```

## Usage

1. Clone the repository to your local machine.

\```bash
git clone https://github.com/Neilus03/simple_GAN.git
\```

2. Navigate to the project directory.

\```bash
cd simple_GAN
\```

3. Run the `simple_gan.py` script.

\```bash
python simple_gan.py
\```

4. To visualize the training process, open TensorBoard with:

\```bash
tensorboard --logdir=runs
\```

## Contributing

Feel free to submit a pull request if you want to make improvements or fix bugs.

## License

MIT License

## Acknowledgments

Thanks to the PyTorch team and the creators of the MNIST dataset.
