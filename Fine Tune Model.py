import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from accelerate import Accelerator
import deepspeed
from accelerate.utils import DummyOptim, DummyScheduler
from tqdm import tqdm

class FineTuner():
    model_checkpoint = ""
    tokenizer_checkpoint = ""

    tokenizer = None
    model = None
    optimizer = None
    lr_scheduler = None

    training_file = ""
    raw_data = None
    dataset = None
    tokenized_dataset = None
    train_dataloader = None
    eval_dataloader = None

    # HYPERPARAMETERS
    input_context_length = 128
    batch_size = 128
    learning_rate = 1e-05
    no_of_epochs = 10
    gradient_accumulation_steps = 1
    save_every = 1

    def __init__(self, model_checkpoint, tokenizer_checkpoint, training_file):
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.training_file = training_file

        self.load_dataset()
        self.process_dataset()
        self.load_tokenizer()
        self.load_model()

    def format_prompt(prompt):
        return prompt["prompt"]

    def tokenize_function(self, input):
        model_inputs = self.tokenizer(
            self.format_prompt(input['prompt']),
            max_length = self.input_context_length, 
            padding = "max_length", 
            truncation = True,
            return_overflowing_tokens = False,
            return_length = False
        )
        model_inputs["labels"] = input['input_ids']
        return model_inputs

    def process_dataset(self):
        self.tokenized_dataset = self.dataset.map(self.tokenize_function, remove_columns=["prompt"])
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.train_dataloader = DataLoader(
            self.tokenized_dataset["train"],
            shuffle = True,
            batch_size = self.batch_size,
            collate_fn = data_collator,
            pin_memory = True
        )
        self.eval_dataloader = DataLoader(
            self.tokenized_dataset["validation"],
            batch_size = self.batch_size,
            collate_fn = data_collator,
            pin_memory = True
        )

    def load_dataset(self):
        with open(self.training_file, "r") as file:
            lines = len(file.readlines())
            range = lines - 1
        file.close()

        print("Total Number of Prompts: ", range)

        self.raw_data = load_dataset("csv", data_files=self.training_file)

        train_test_split = self.raw_data["train"].train_test_split(test_size=0.05, shuffle=True, seed=42)
        train_validation_split = train_test_split["train"].train_test_split(test_size=0.05, shuffle=True, seed=42)

        train_set = train_validation_split["train"]
        validation_set = train_validation_split["test"]
        test_set = train_test_split["test"]

        self.dataset = {"train": train_set, "validation": validation_set, "test": test_set}

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_checkpoint,
            add_eos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loaded Tokenizer: ", self.tokenizer)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_checkpoint, 
            trust_remote_code = True,
            torch_dtype = torch.bfloat16
        )

        print("Loaded Model: ", self.model)

    def save_model(self):
        

    def fine_tune(self):
        torch.backends.cuda.matmul.allow_tf32 = True

        accelerator = Accelerator(split_batches = True)
        device = accelerator.device

        self.model.train()
        self.model.config.pad_token_id = self.tokenzer.pad_token_id

        optimizer_class = (
            deepspeed.ops.adam.FusedAdam
            if accelerator.state.deepspeed_plugin is None or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )

        self.optimizer = optimizer_class(self.model.parameters(), lr = self.learning_rate)

        num_training_steps = self.no_of_epochs * len(self.train_dataloader) // self.gradient_accumulation_steps

        if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.lr_scheduler = get_scheduler(
                name="linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
        else:
            self.lr_scheduler = DummyScheduler(
                self.optimizer, 
                total_num_steps=num_training_steps, 
                warmup_num_steps=0
            )

        self.train_dataloader, self.eval_dataloader, self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.train_dataloader,
            self.eval_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler
        )

        progress_bar = tqdm(range(num_training_steps))

        self.model.gradient_checkpointing_enable() 

        for epoch in range(self.no_of_epochs):
            self.model.train()
            training_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                training_loss += loss.detach().float()
                loss = loss / self.gradient_accumulation_steps
                accelerator.backward(loss)

                if step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    progress_bar.update(1)

            self.model.eval()
            evaluation_loss = 0

            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)

                loss = outputs.loss
                eval_loss += loss.detach().float()

            accelerator.print(f"epoch: {epoch}")
            accelerator.print(f"Training Loss: {training_loss}")
            accelerator.print(f"Evaluation Loss: {evaluation_loss}")

            if epoch & self.save_every == (self.save_every - 1):
                self.save_model()

            
