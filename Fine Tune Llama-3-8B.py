import argparse
import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler
from tqdm.auto import tqdm
from accelerate.utils import DummyOptim, DummyScheduler
import deepspeed


## HYPERPARAMETERS
default_batch_size = 256
default_epochs = 10
default_lr = 4e-05
save_every = 1
context_length = 128
num_proc = 4
weight_decay = 0.01
temperature = 0.01
top_p = 1
num_return_sequences = 1
gradient_accumulation_steps = 1


## MODEL DETAILS
save_directory_path = ""
log_file = "training_logs.txt"
log_file_path = save_directory_path + "/" + log_file
model_name = "llama_3_8b"
base_model_checkpoint = "meta-llama/Meta-Llama-3-8B"
tokenizer_checkpoint = "meta-llama/Meta-Llama-3-8B"


parser = argparse.ArgumentParser(description="arg-parser for fine tuning llama")
# Add optional argument with default value
parser.add_argument(
    "-epochs",
    "--num_epochs",
    type=int,
    default=default_epochs,
    help="Number of epochs to run",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=default_lr,
    help="Learning rate",
)
parser.add_argument(
    "-data",
    "--datatset_file",
    help="Training data file path",
)
parser.add_argument(
    "-save_every",
    "--save_every",
    type=int,
    default=save_every,
    help="Save After Every Nth Epoch",
)
parser.add_argument(
    "-batch",
    "--batch_size",
    type=int,
    default=default_batch_size,
    help="Batch size for training",
)
# Parse the command-line arguments
args = parser.parse_args()


num_of_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate


def write_log_to_file(log, log2=""):
    print(log)
    if len(log2) > 0:
        print(log2)
    with open(log_file_path, "a") as file:
        log_str = log
        if isinstance(log, list):
            log_str = ", ".join(str(item) for item in log)
        log2_str = log2
        if isinstance(log2, list):
            log2_str = ", ".join(str(item) for item in log2)
        file.write(log_str)
        file.write(log2_str)
        file.write("\n")


write_log_to_file(f"### Hyperparameters ###")
write_log_to_file(f"Epoch -> {num_of_epochs}")
write_log_to_file(f"Learning Rate -> {learning_rate}")
write_log_to_file(f"Batch Size -> {batch_size}")


# enables the use of TensorFloat-32 (TF32) precision for matrix multiplication operations on NVIDIA GPUs when using PyTorch.
torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token  # end of sequence token

generate_tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_checkpoint, padding_side="left"
)
generate_tokenizer.pad_token = generate_tokenizer.eos_token

print("Tokenizer:- ", tokenizer)


training_file = ""  # csv file with only prompt column

with open(training_file, "r") as fp:
    lines = len(fp.readlines())
    rng = lines - 1
    print("Total Number of lines :", lines)
fp.close()
print("Total Number of prompts :", rng)

raw_data = load_dataset("csv", data_files=training_file)

ds = (
    raw_data["train"]
    .shuffle(seed=42)
    .select(range(rng))
    .train_test_split(test_size=0.05, shuffle=True, seed=42)
)

train_eval_ds = ds["train"].train_test_split(test_size=0.05, shuffle=True, seed=42)
ds["train"] = train_eval_ds["train"]
ds["validation"] = train_eval_ds["test"]


def format_prompt(prompt):
    return prompt["prompt"]


def tokenize_function(examples):
    return tokenizer(
        format_prompt(examples),
        padding="max_length",
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=False,
    )


tokenized_datasets = ds.map(tokenize_function, remove_columns=["prompt"])


def group_texts(examples):
    result = examples.copy()
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, remove_columns=[])


# collating (combining) the data into batches, handles padding so that each sequence in the batch has the same length for efficient training.
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Load datasets and batch them for training or evaluation
train_dataloader = DataLoader(
    lm_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
    pin_memory=True,
)
eval_dataloader = DataLoader(
    lm_datasets["validation"],
    batch_size=batch_size,
    collate_fn=data_collator,
    pin_memory=True,
)


test_prompts = ["sample prompts to test output during training"]


def fine_tune_model():

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        base_model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    model.train()
    model.config.pad_token_id = tokenizer.pad_token_id

    optimizer_cls = (
        deepspeed.ops.adam.FusedAdam
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    num_training_steps = (
        num_of_epochs * len(train_dataloader)
    ) // gradient_accumulation_steps

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=num_training_steps, warmup_num_steps=0
        )

    train_dl, eval_dl, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, lr_scheduler
    )

    progress_bar = tqdm(range(num_training_steps))

    model.gradient_checkpointing_enable()

    test_input_ids = generate_tokenizer(test_prompts, return_tensors="pt", padding=True)
    test_device_input_ids = {k: v.to(device) for k, v in test_input_ids.items()}

    for epoch in range(num_of_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dl):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        model.eval()
        total_loss_eval = 0
        for step, batch in enumerate(eval_dl):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            total_loss_eval += loss.detach().float()

        predictions = model.generate(
            **test_device_input_ids,
            max_new_tokens=50,
            do_sample=False,
            num_return_sequences=4,
            num_beams=4,
        )

        predictions = [
            generate_tokenizer.decode(x, skip_special_tokens=True) for x in predictions
        ]

        if accelerator.is_main_process:
            write_log_to_file(f"epoch {epoch}:")
            for idx, entry in enumerate(test_prompts):
                write_log_to_file("Input Ids: ", entry)
                write_log_to_file("Predictions: ", predictions[idx * 4 : (idx + 1) * 4])
                accelerator.print("Input Ids: ", entry)
                accelerator.print(
                    "Predictions: ", predictions[idx * 4 : (idx + 1) * 4], "\n"
                )

            # eval_metric = metric.compute()
            accelerator.print(f"epoch {epoch}:")
            accelerator.print(f"Loss: {total_loss};  Eval_Loss: {total_loss_eval}")
            write_log_to_file(f"epoch {epoch}:")
            write_log_to_file(f"Loss: {total_loss};  Eval_Loss: {total_loss_eval}")

        if epoch % args.save_every == (args.save_every - 1):
            accelerator.print("saving model")
            version = (
                training_file
                + str(epoch)
                + "_"
                + str(batch_size)
                + "_"
                + str(learning_rate)
                + "_"
                + str(gradient_accumulation_steps)
            )

            accelerator.wait_for_everyone()
            save_path_name = (
                save_directory_path + "/" + "fine_tuned_" + model_name + "_" + version
            )
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                save_path_name,
                save_function=accelerator.save,
                is_main_process=accelerator.is_main_process,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
            )

            accelerator.print("model saved at ", save_path_name)
            tokenizer.save_pretrained(save_path_name)


if __name__ == "__main__":
    fine_tune_model()
