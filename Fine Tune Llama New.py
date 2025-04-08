# hyperparameters
batch_size = 128
learning_rate = 5e-06
gradient_accumulation_steps = 2
epochs = 10
save_every = 1
context_length = 128


# base model
model_name = "llama_3_8b"
model_checkpoint = "meta-llama/Meta-Llama-3-8B"
tokenizer_checkpoint = "meta-llama/Meta-Llama-3-8B"
save_directory = "fine_tune_3_Nov"
log_file = f"{save_directory}/{model_name}_training_logs.txt"


# training data
data_directory = "Raw_Data"
data_file_name = "prompts_rpm_data_15_08-02_09.csv"
data_file = f"{data_directory}/{data_file_name}"


def format_log(log):
    if isinstance(log, list):
        return ", ".join(str(item) for item in log)
    return str(log)


def write_log_to_file(log1, log2=""):
    print(log1)
    if log2:
        print(log2)

    log1_str = format_log(log1)
    log2_str = format_log(log2)

    with open(log_file, "a") as file:
        file.write(log1_str)
        file.write("\n")
        if log2_str:
            file.write(log2_str)
            file.write("\n")


write_log_to_file("##### HYPERPARAMETERS #####")
write_log_to_file(f"batch size: {batch_size}")
write_log_to_file(f"learning rate: {learning_rate}")
write_log_to_file(f"gradient accumulation steps: {gradient_accumulation_steps}")
write_log_to_file(f"number of epochs: {epochs}")
write_log_to_file(f"save every: {save_every}")


with open(data_file, "r") as fp:
    num_prompts = len(fp.readlines()) - 1
    write_log_to_file(f"Total Number of Prompts: , {num_prompts}")
fp.close()


import torch
import os
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    get_scheduler,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import deepspeed
from tqdm.auto import tqdm
import traceback
from accelerate.utils import DummyOptim, DummyScheduler, DeepSpeedPlugin


torch.backends.cuda.matmul.allow_tf32 = True


# initialize tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pad_token = AutoTokenizer.from_pretrained(tokenizer_checkpoint).eos_token

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_checkpoint,
    padding_side="left",
    pad_token=pad_token,
)

generate_tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_checkpoint,
    padding_side="left",
    pad_token=pad_token,
)

write_log_to_file(f"Tokenizer: {tokenizer}")


# load data
raw_data = load_dataset("csv", data_files=data_file)

ds = (
    raw_data["train"]
    .shuffle(seed=100)
    .select(range(num_prompts))
    .train_test_split(test_size=0.05, shuffle=True, seed=100)
)

train_eval_ds = ds["train"].train_test_split(test_size=0.05, shuffle=True, seed=100)
ds["train"], ds["validation"] = train_eval_ds["train"], train_eval_ds["test"]


def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=False,
    )


tokenized_datasets = ds.map(tokenize_function, remove_columns=["prompt"])


def set_labels(input):
    input["labels"] = input["input_ids"].copy()
    return input


lm_datasets = tokenized_datasets.map(set_labels, batched=True)

data_collater = DataCollatorForLanguageModeling(tokenizer, mlm=False)

train_dataloader = DataLoader(
    lm_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collater,
    pin_memory=True,
)

eval_dataloader = DataLoader(
    lm_datasets["validation"],
    batch_size=batch_size,
    collate_fn=data_collater,
    pin_memory=True,
)

write_log_to_file(f"file format is: {ds['train'][0]}")


t_urls = [
    "https://www.example.com/0.6",
    "https://www.example.com/1.6",
    "https://www.example.com/0.2",
    "<url>",
    "https://www.jdpower.com",
    "https://www.housinglist.com/hud/fl",
    "https://usafrugalclub.com/",
    "https://www.urbandictionary.com/author.php/",
    "https://www.hireteen.com/summer-jobs/",
    "https://www.austedo.com/Austedo",
    "https://www.medicine.com/cipla",
    "https://www.gsk.com/gsk",
    "https://www.forbes.com/sites/hughmcintyre/2023/09/15/jimmy-buffett-charts-his-first-no-1-song-posthumously/",
    "https://www.forbes.com/sites/hughmcintyre/2023/09/15/jimmy-charts-his-first-no-1-song-posthumously/",
    "https://www.healthline.com/Anemia & Hemophilia Signs",
    "https://www.healthline.com/Best Lion's Mane Mushroom Supplements",
    "https://www.thestreet.com/Southwest Airline Deals For Seniors",
]

test_urls = []

for url in t_urls:
    test_urls.append(f"The url is {url}. The advertising search term is")


def main():
    accelerator = Accelerator(split_batches=True, device_placement=True, deepspeed_plugin=DeepSpeedPlugin())
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    model.train()
    model.gradient_checkpointing_enable()

    model.config.pad_token_id = tokenizer.pad_token_id

    optimizer_cls = (
        deepspeed.ops.adam.FusedAdam
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    num_training_steps = epochs * len(train_dataloader) // gradient_accumulation_steps

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

    train_loader, eval_loader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, lr_scheduler
    )

    progress_bar = tqdm(range(num_training_steps))

    test_input_ids = tokenizer(test_urls, return_tensors="pt", padding=True)
    test_device_input_ids = {k: v.to(device) for k, v in test_input_ids.items()}

    for epoch in range(epochs):
        model.train()
        total_loss, total_adjusted_loss = 0, 0

        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss

            generated_output = model.generate(
                **batch,
                max_new_tokens=25,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            scale_factor = 0.01
            original_texts = [
                generate_tokenizer.decode(input_ids, skip_special_tokens=True)
                for input_ids in batch["input_ids"]
            ]
            generated_texts = [
                generate_tokenizer.decode(output, skip_special_tokens=True)
                for output in generated_output
            ]

            new_token_lengths = [
                len(gen_text) - len(orig_text)
                for gen_text, orig_text in zip(generated_texts, original_texts)
            ]

            penalties = [
                max(0, new_len - 9) + max(0, 3 - new_len)
                for new_len in new_token_lengths
            ]
            length_penalty = (
                torch.tensor(penalties, device=loss.device).float().mean()
                * scale_factor
            )

            adjusted_loss = (loss + length_penalty) / gradient_accumulation_steps

            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            total_loss += loss.detach()
            total_adjusted_loss += adjusted_loss.detach()

        model.eval()
        total_eval_loss = 0

        with torch.no_grad():
            for step, batch in enumerate(eval_loader):
                outputs = model(**batch)
                loss = outputs.loss
                total_eval_loss += loss.detach()

        predictions = model.generate(
            **test_device_input_ids,
            max_new_tokens=50,
            do_sample=False,
            num_return_sequences=4,
            num_beams=4,
        )

        decoded_predictions = [
            generate_tokenizer.decode(x, skip_special_tokens=True) for x in predictions
        ]

        if accelerator.is_main_process:
            write_log_to_file(f"epoch: {epoch}")
            for idx, entry in enumerate(test_urls):
                write_log_to_file(f"Input Ids: ")
                write_log_to_file(
                    f"Prediction: ", decoded_predictions[idx * 4 : (idx + 1) * 4]
                )

            write_log_to_file(f"epoch: {epoch}")
            write_log_to_file(
                f"Loss: {total_loss};  Adjusted_Loss: {total_adjusted_loss};  Eval_Loss: {total_eval_loss}"
            )

        if epoch % save_every == save_every - 1:
            write_log_to_file("saving model")
            version = (
                data_file
                + str(epoch)
                + "_"
                + str(batch_size)
                + "_"
                + str(learning_rate)
                + "_"
                + str(gradient_accumulation_steps)
            )

        accelerator.wait_for_everyone()
        save_path = save_directory + "/" + model_name + "_" + version
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path,
            save_function=accelerator.save,
            is_main_process=accelerator.is_main_process,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=False,
        )

        tokenizer.save_pretrained(save_path)

        write_log_to_file(f"model saved at {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_log_to_file(e)
        traceback.print_exc()
