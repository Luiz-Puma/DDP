import os
import argparse
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch.distributed as dist
import logging


# Create file logger
def create_logger(log_file, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(log_level if rank == 0 else 'ERROR')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

# Argument parsing function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Distributed training setup.")
    parser.add_argument("--output_dir", type=str, default="training_output", help="Directory to save outputs")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--chunks", type=int, default=4, help="Number of micro-batches (chunks) to pipeline per global batch.")
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
    def __init__(self, local_rank, rank, world_size, model, schedule, train_data, optimizer, output_dir, num_epochs,
                 save_every, report_rate, max_grad_norm):

        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(local_rank)
        self.schedule = schedule
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
        self.logger = create_logger(self.output_dir + '/log.txt')

    def save_checkpoint(self, epoch):
        """Save model checkpoint (only by the main process)."""
        if self.local_rank == 0:
            save_path = os.path.join(self.output_dir, f"rank_{self.rank}_checkpoint_epoch_{epoch}.pt")
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"[GPU {self.local_rank}] Checkpoint saved at epoch {epoch} to {save_path}")

    def train_step(self, batch):
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        batch = {k: v.to(self.local_rank) for k, v in batch.items() if k != 'labels' or self.rank == self.world_size - 1}
        if self.rank == 0 or self.rank == self.world_size - 1:
            outputs = self.schedule.step(**batch)
        else:
            self.schedule.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        loss = torch.tensor(0.0, device=self.local_rank)
        if self.rank == self.world_size - 1:
            loss = outputs.loss.detach()
        dist.broadcast(loss, src=self.world_size - 1)
        return loss.item()

    def train(self):
        """Run the full training loop"""
        for epoch in range(self.num_epochs):
            self.train_data.sampler.set_epoch(epoch)
            sum_loss = 0.0
            
            self.logger.info(f"[GPU {self.local_rank}] | Epoch {epoch}/{self.num_epochs}")
            for step, batch in enumerate(self.train_data):
                self.logger.info((f"Rank {self.rank}" 
                                  f"; input_ids shape {batch['input_ids'].shape}"
                                  f"; attention_mask shape {batch['attention_mask'].shape}"
                                  f"; label shape {batch['labels'].shape}"))
                loss = self.train_step(batch)
                sum_loss += loss

                if step != 0 and step % self.report_rate == 0:
                    avg_loss = sum_loss / self.report_rate
                    self.logger.info(f"[GPU {self.local_rank}] | Epoch {epoch}/{self.num_epochs} |"
                          f"Step {step} | Loss: {avg_loss:.4f}")

                    # global report: reporting the average loss across all GPUs
                    avg_loss = torch.Tensor([avg_loss]).to(self.local_rank)
                    # this will get the sum of the tensor from all GPUs, and put result only on GPU 0
                    dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.SUM)
                    if self.local_rank == 0:
                        all_gpus_avg_loss = avg_loss / self.world_size
                        self.logger.info(f"All_GPUs_Loss: {all_gpus_avg_loss.item():.4f}")
                    # reset the loss
                    sum_loss = 0.0

            # save checkpoint
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self.save_checkpoint(epoch + 1)

def prepare_dataset(batch_size):
    # Load a subset of the C4 dataset with a glob pattern for specific training files
    dataset = load_dataset("allenai/c4", data_files=["multilingual/c4-af.tfrecord-00000-of-00064.json.gz"])

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
    local_rank=int(os.environ["LOCAL_RANK"])
    rank=int(os.environ['RANK'])
    world_size=int(os.environ["WORLD_SIZE"])
    
    # setup distributed data parallel
    ddp_setup()

    # Initialize model
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config)    
    
    # Split model
    seq_length = model.config.n_positions
    decoders_per_rank = (model.config.n_layer + world_size - 1) // world_size
    split_spec = {f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING 
                  for i in range(1, world_size)}
    micro_batch_size = args.batch_size // args.chunks
    input_ids = torch.randint(0, model.config.vocab_size, (micro_batch_size, seq_length))
    labels = torch.randint(0, model.config.vocab_size, (micro_batch_size, seq_length))
    attention_mask = torch.randint(0, model.config.vocab_size, (micro_batch_size, seq_length))
    mb_inputs = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        
    pipe = pipeline(model, mb_args=(), mb_kwargs=mb_inputs, split_spec=split_spec)
    assert pipe.num_stages == world_size
    stage_model = pipe.get_stage_module(rank)
    
    # Create schedule runtime
    stage = pipe.build_stage(rank, device=torch.device(f"cuda:{local_rank}"))
    schedule = ScheduleGPipe(stage, args.chunks)

    # Prepare the dataset and optimizer
    train_data_loader = prepare_dataset(batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(stage_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Instantiate Trainer and start training
    trainer = Trainer(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        model=stage_model,
        schedule=schedule,
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