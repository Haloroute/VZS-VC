# PyTorch Dataset and DataLoader definitions
import datasets

from torch.utils.data import IterableDataset
from typing import Iterator, Literal


# Dataset class for VieNeu-TTS-140h, using Hugging Face's datasets library in streaming mode
class VieNeuTTSDataset(IterableDataset):
    def __init__(
        self, 
        name: str,
        sampling_rate: int = 24000,
        split: str = "train", 
        part: Literal["train", "val"] = "train", 
        val_size: int | float = 0.1, 
        seed: int | None = None
    ):
        super().__init__()
        # Use provided name
        self.dataset_name = name
        self.sampling_rate = sampling_rate
        self.split = split # VieNeu-TTS-140h has only 'train' split, so we will handle val/test splitting ourselves
        self.part = part

        # Load the dataset using Hugging Face's datasets library. Use stream mode for online datasets
        full_dataset = datasets.load_dataset(self.dataset_name, split=self.split)

        # Extract total length from Hugging Face metadata
        total_len = full_dataset.info.splits[self.split].num_examples

        # Calculate split length based on whether val_size is a ratio or an absolute number
        if isinstance(val_size, float):
            val_len = int(total_len * val_size)
        else:
            val_len = val_size
        
        # Since train_test_split doesn't work out-of-the-box on IterableDataset,
        # we shuffle with a buffer and use take()/skip() to split.
        if seed is not None:
            full_dataset = full_dataset.shuffle(seed=seed)
        
        # Use take() for validation split and skip() for training split
        if self.part == "val":
            self._len = val_len
            self.dataset = full_dataset.take(self._len)
        elif self.part == "train":
            self._len = total_len - val_len
            self.dataset = full_dataset.skip(val_len)
        else:
            raise ValueError(f"Invalid part: {self.part}. Must be 'train' or 'val'.")

    # Implement __len__ and __iter__ for IterableDataset
    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[dict]:
        for item in self.dataset:
            try:
                # audio_bytes = item.get("audio", {}).get("bytes", None)
                # audio_tensor, orig_sr = torchaudio.load(io.BytesIO(audio_bytes))

                # # Resample if needed
                # if orig_sr and orig_sr != self.sampling_rate:
                #     resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sampling_rate)
                #     audio_tensor = resampler(audio_tensor)
                    
                # yield TensorDict({
                #     "audio": audio_tensor,
                #     "speaker": NonTensorData(speaker),
                #     "text": NonTensorData(text),
                # })
                # yield audio tensor plus all other metadata fields
                # yield {"audio": audio_tensor, **item}
                yield item
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
