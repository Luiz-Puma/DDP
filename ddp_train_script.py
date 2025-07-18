import os
import argparse
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch.distributed as dist

# Argument parsing function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Distributed training setup.")
    parser.add_argument("--output_dir", type=str, default="training_output", help="Directory to save outputs")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every N epochs")
    parser.add_argument("--report_rate", type=int, default=10, help="Report rate every N steps")
    return parser.parse_args()

# DDP setup function
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(self, local_rank, rank, world_size, model, train_data, optimizer, output_dir, num_epochs,
                 save_every, report_rate, max_grad_norm):

        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(local_rank)
        self.model = DDP(self.model, device_ids=[local_rank])
        self.train_data = train_data
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.report_rate = report_rate
        self.local_rank = local_rank
        self.max_grad_norm = max_grad_norm

        if self.local_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        """Save model checkpoint (only by the main process)."""
        if self.local_rank == 0:
            save_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), save_path)
            print(f"[GPU {self.local_rank}] Checkpoint saved at epoch {epoch} to {save_path}")

    def train_step(self, batch):
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        batch = {k: v.to(self.local_rank) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def train(self):
        """Run the full training loop"""
        for epoch in range(self.num_epochs):
            self.train_data.sampler.set_epoch(epoch)
            sum_loss = 0.0

            for step, batch in enumerate(self.train_data):
                loss = self.train_step(batch)
                sum_loss += loss

                if step != 0 and step % self.report_rate == 0:
                    avg_loss = sum_loss / self.report_rate
                    print(f"[GPU {self.local_rank}] | Epoch {epoch}/{self.num_epochs} |"
                          f"Step {step} | Loss: {avg_loss:.4f}")

                    # global report: reporting the average loss across all GPUs
                    avg_loss = torch.Tensor([avg_loss]).to(self.local_rank)
                    # this will get the sum of the tensor from all GPUs, and put result only on GPU 0
                    dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.SUM)
                    if self.local_rank == 0:
                        all_gpus_avg_loss = avg_loss / self.world_size
                        print(f"All_GPUs_Loss: {all_gpus_avg_loss.item():.4f}")
                    # reset the loss
                    sum_loss = 0.0

            # save checkpoint
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

def prepare_dataset(batch_size):
    # Load a subset of the C4 dataset with a glob pattern for specific training files
    dataset = load_dataset("allenai/c4", data_files=["multilingual/c4-af.tfrecord-00000-of-00064.json.gz"], trust_remote_code=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def collate_fn(batch):
        """
        Custom collate function to tokenize each batch dynamically during data loading.
        """
        # Extract text data
        texts = [item["text"] for item in batch]

        # Set the pad token if it isn't set already
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize text data
        encoding = tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt")
        encoding["labels"] = encoding["input_ids"].clone()  # Use `input_ids` as labels

        # Return tokenized input tensors
        return encoding

    # Create DistributedSampler for proper shuffling and partitioning across processes
    dist_sampler = DistributedSampler(dataset["train"], shuffle=True)

    # Create DataLoader with custom collate_fn
    data_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        sampler=dist_sampler,
        collate_fn=collate_fn,
        drop_last=True
    )

    return data_loader

def main():
    args = parse_arguments()

    # setup distributed data parallel
    ddp_setup()

    # Initialize model
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config)

    # Prepare the dataset and optimizer
    train_data_loader = prepare_dataset(batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Instantiate Trainer and start training
    trainer = Trainer(
        local_rank=int(os.environ["LOCAL_RANK"]),
        rank=int(os.environ['RANK']),
        world_size=int(os.environ["WORLD_SIZE"]),
        model=model,
        train_data=train_data_loader,
        optimizer=optimizer,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        report_rate=args.report_rate,
        max_grad_norm=args.max_grad_norm
    )
    trainer.train()

    # terminate the ddp
    destroy_process_group()

if __name__ == "__main__":
    main()