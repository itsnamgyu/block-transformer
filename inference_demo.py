import argparse
import glob
import os
import time

import torch
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from transformers.generation.streamers import BaseStreamer

from model.block_transformer import BlockTransformer
from model.utils import load_block_transformer_from_config, load_vanilla_model_from_config
from paths import PROJECT_ROOT
from util.config import load_config
from util.tokenizer import TOKENIZERS


def get_config_path(name):
    if ".yaml" not in name:
        name += ".yaml"
    return os.path.join(PROJECT_ROOT, "conf", "trainer", name)


def get_checkpoint_path(name, checkpoint_root):
    root = os.path.join(checkpoint_root, name)
    if not os.path.exists(root):
        raise ValueError(f"Checkpoint directory does not exist: {root}")
    pattern = os.path.join(checkpoint_root, name, "checkpoint-*")
    checkpoint_paths = glob.glob(pattern)

    def get_step(checkpoint_path):
        bs = os.path.basename(checkpoint_path)
        return int(bs.split("-")[1])

    # final checkpoint is the one with the highest step
    checkpoint_paths = [(get_step(cp), cp) for cp in checkpoint_paths]
    checkpoint_paths.sort()
    checkpoint_path = checkpoint_paths[-1][1]
    checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")
    print(f"Retrieving latest checkpoint path {checkpoint_path}")
    return checkpoint_path


def load_model(name, checkpoint_root, block=True):
    config = load_config(get_config_path(name))
    with init_empty_weights():
        if block:
            model, tokenizer = load_block_transformer_from_config(config)
        else:
            model = load_vanilla_model_from_config(config)
    checkpoint = get_checkpoint_path(name, checkpoint_root)
    device_map = "sequential"  # set to auto to use multiple GPUs + pipelining (not tested)
    model = load_checkpoint_and_dispatch(model, checkpoint=checkpoint, device_map=device_map)
    if not block and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    if block:
        return model, tokenizer
    else:
        return model


def set_temperature(model, temperature):
    if isinstance(model, BlockTransformer):
        model.token_decoder.generation_config.update(do_sample=True, temperature=temperature)
    else:
        model.generation_config.update(do_sample=True, temperature=temperature)


class FirstSampleStreamer(BaseStreamer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.first = True

    def put(self, value):
        # ignore prompt
        if self.first:
            self.first = False
            return
        token = self.tokenizer.decode(value[-1]).replace("\n", "\\n")
        print(token, end="", flush=True)

    def end(self):
        self.first = True


def get_model_display_name(model_key):
    if "vanilla" in model_key:
        return f"Vanilla {model_key.split('_')[-1].upper()} model"
    elif "block" in model_key:
        return f"Block Transformer {model_key.split('_')[-1].upper()}"
    else:
        raise ValueError(f"Unsupported model key: {model_key}")


def main(args):
    model_display_name = get_model_display_name(args.model)
    print(f" Loading {model_display_name} ".center(60, "-"))
    if "vanilla" in args.model:
        model = load_model(args.model, args.checkpoint_root, block=False)
        tokenizer = TOKENIZERS["pythia"]
    else:
        model, tokenizer = load_model(args.model, args.checkpoint_root, block=True)

    prompt = """It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.
    
    There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France. In both countries it was clearer than crystal to the lords of the State preserves of loaves and fishes, that things in general were settled for ever.
    
    It was the year of Our Lord one thousand seven hundred and seventy-five. Spiritual revelations were conceded to England at that favoured period, as at this. Mrs. Southcott had recently attained her five-and-twentieth blessed birthday, of whom a prophetic private in the Life Guards had heralded the sublime appearance by announcing that arrangements were made for the swallowing up of London and Westminster. Even the Cock-lane ghost had been laid only a round dozen of years, after rapping out its messages, as the spirits of this very year last past (supernaturally deficient in originality) rapped out theirs. Mere messages in the earthly order of events had lately come to the En- glish Crown and People, from a congress of British subjects in America: which, strange to relate, have proved more important to the human race than any communications yet received through any of the chickens of the Cock-lane brood.
    
    France, less favoured on the whole as to matters spiritual than her sister of the shield and trident, rolled with exceeding smoothness down hill, making paper money and spending it. Under the guidance of her Christian pastors, she entertained herself, besides, with such humane achievements as sentencing a youth to have his hands cut off, his tongue torn out with pincers, and his body burned alive, because he had not kneeled down in the rain to do honour to a dirty procession of monks which passed within his view, at a distance of some fifty or sixty yards. It is likely enough that, rooted in the woods of France and Norway, there were growing trees, when that sufferer was put to death, already marked by the Woodman, Fate, to come down and be sawn into boards, to make a certain movable framework with a sack and a knife in it, terrible in history. It is likely enough that in the rough outhouses of some tillers of the heavy lands adjacent to Paris, there were sheltered from the weather that very day, rude carts, bespattered with rustic mire, snuffed about by pigs, and roosted in by poultry, which the Farmer, Death, had already set apart to be his tumbrils of the Revolution. But that Woodman and that Farmer, though they work unceasingly, work silently, and no one heard them as they went about with muffled tread: the rather, forasmuch as to entertain any suspicion that they were awake, was to be atheistical and traitorous.

    In England, there was scarcely an amount of order and protection to justify much national boasting. Daring burglaries by armed men, and highway robberies, took place in the capital itself every night; families were publicly cautioned not to go out of town without removing their furniture to upholsterers’ warehouses for security; the highwayman in the dark was a City tradesman in the light, and, being recognised and challenged by his fellow-tradesman whom he stopped in his character of “the Captain,” gallantly shot him through the head and rode away; the mall was waylaid by seven robbers, and the guard shot three dead, and then got shot dead himself by the other four, “in consequence of the fail- ure of his ammunition:” after which the mall was robbed in peace; that magnificent potentate, the Lord Mayor of London, was made to stand and deliver on Turnham Green, by one highwayman, who despoiled the illustrious creature in sight of all his retinue; prisoners in London gaols fought battles with their turnkeys, and the majesty of the law fired blun- derbusses in among them, loaded with rounds of shot and ball; thieves snipped off diamond crosses from the necks of noble lords at Court drawing-rooms; musketeers went into St. Giles’s, to search for contra- band goods, and the mob fired on the musketeers, and the musketeers fired on the mob, and nobody thought any of these occurrences much out of the common way.
    
    """
    prompt_length = len(tokenizer(prompt)["input_ids"])
    print(" Generation parameters ".center(60, "-"))
    generation_length = args.max_length - prompt_length
    print(f"Prompt length              : {prompt_length:>8}")
    print(f"Generation length          : {generation_length:>8}")
    print(f"Batch size                 : {args.batch_size:>8}")

    print(" Start of Prompt ".center(60, "-"))
    print(prompt[:500])
    print(" End of Prompt ".center(60, "-"))
    print(prompt[-500:])

    set_temperature(model, temperature=args.temperature)
    prompts = [prompt] * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt")
    inputs = {k: t.cuda() for k, t in inputs.items()}
    print("Synchronizing time... ", end="", flush=True)
    current = time.time()
    time_to_next_slot = 5 - current % 5
    time.sleep(time_to_next_slot)
    for i in range(3, 0, -1):
        print(f"{i} ", end="", flush=True)
        time.sleep(1)
    print()
    print(" Start of generation ".center(60, "-"))
    start = time.time()
    if isinstance(model, BlockTransformer):
        streamer = FirstSampleStreamer(tokenizer)
        output_ids = model.generate(**inputs, max_length=args.max_length, streamer=streamer)
    else:
        streamer = FirstSampleStreamer(tokenizer)
        output_ids = model.generate(**inputs, max_length=args.max_length, streamer=streamer)

    duration = time.time() - start
    memory = torch.cuda.max_memory_allocated(device=None)
    print()
    print()
    print(" End of generation ".center(60, "-"))
    print(f"Model                      : {model_display_name}")
    print("-" * 60)
    print(f"Prompt length              : {prompt_length:>8}")
    print(f"Generation length          : {generation_length:>8}")
    print(f"Batch size                 : {args.batch_size:>8}")
    print("-" * 60)
    print(f"Max memory allocated       : {memory / 1024 ** 3:>6.2f}GB")
    print(f"Tok/sec/sample             : {generation_length / duration:>8.2f}")
    print(f"Tok/sec                    : {generation_length * args.batch_size / duration:>8.2f}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--max_length", type=int, default=1334, help="Maximum length for generation")
    parser.add_argument("--model", type=str, default="block_main_b4_1.2b",
                        help="Model to use for generation (e.g., vanilla_410, block_main_b4_1.2b)")
    parser.add_argument("--checkpoint_root", type=str, default="results",
                        help="Root directory for model checkpoints")
    args = parser.parse_args()

    main(args)
