from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

class TensorBoardLogger:
    def __init__(self, log_dir):
        """
        Initialize the TensorBoard logger.
        Args:
            log_dir (str): Directory where TensorBoard logs will be saved.
        """
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """
        Log a scalar value.
        Args:
            tag (str): Name of the scalar.
            value (float): Value to log.
            step (int): Training step or epoch.
        """
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, images, step, normalize=True):
        """
        Log a batch of images.
        Args:
            tag (str): Name of the image batch.
            images (torch.Tensor): Batch of images to log (shape: [N, C, H, W]).
            step (int): Training step or epoch.
            normalize (bool): Whether to normalize images to [0, 1].
        """
        self.writer.add_images(tag, images, step, normalize=normalize)

    def log_histogram(self, tag, values, step):
        """
        Log a histogram of values.
        Args:
            tag (str): Name of the histogram.
            values (torch.Tensor): Values to log.
            step (int): Training step or epoch.
        """
        self.writer.add_histogram(tag, values, step)

    def log_embeddings(self, embeddings, metadata, tag="embeddings"):
        """
        Log embeddings for visualization in TensorBoard.
        Args:
            embeddings (torch.Tensor): Embedding vectors (shape: [N, D]).
            metadata (list): Metadata for each embedding (e.g., labels).
            tag (str): Name of the embedding set.
        """
        self.writer.add_embedding(embeddings, metadata, tag=tag)

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()