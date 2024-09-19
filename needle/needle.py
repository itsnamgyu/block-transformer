import json
import random
import glob
import os
import argparse
from paths import PROJECT_ROOT
from transformers import AutoTokenizer
from tqdm import tqdm

RANDOM_NEEDLE_CITIES = [
    'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
    'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
    'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
    'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
    'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
    'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
    'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
    'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
    'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
    'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
    'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
    'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
]


def generate_random_number(num_digits):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10 ** num_digits - 1
    return random.randint(lower_bound, upper_bound)


def generate_samples(n: int, max_length: int, haystack_dir: str, tokenizer, depth_percents: list, output_dir: str, prompt_type: str):
    os.makedirs(output_dir, exist_ok=True)
    haystack_pile = read_haystack_pile(haystack_dir)
    haystack_pile_tokens = tokenizer.encode(haystack_pile)

    for depth_percent in depth_percents:
        samples = []
        for _ in tqdm(range(n), desc=f"Generating Samples for Depth {depth_percent}%"):
            random_city = random.choice(RANDOM_NEEDLE_CITIES)
            needle_number = str(generate_random_number(7))
            needle = f"The special magic {random_city} number is: {needle_number}"
            context = generate_context(haystack_pile_tokens, tokenizer, max_length, needle, depth_percent)
            question = f"What is the special magic {random_city} number?"

            prompt = construct_prompt(context, question, random_city, prompt_type)

            sample = {
                "city": random_city,
                "needle_number": needle_number,
                "needle": needle,
                "context": context,
                "context_length": max_length,
                "prompt": prompt,
                "answer": needle_number
            }
            samples.append(sample)

        filename = f"samples_depth_{depth_percent}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(samples, f)


def read_haystack_pile(haystack_dir):
    haystack_pile = ""   # concatenation of all haystack files
    haystack_dir = f"{PROJECT_ROOT}/{haystack_dir}" if not os.path.isabs(haystack_dir) else haystack_dir
    for file in glob.glob(f"{haystack_dir}/*.txt"):
        with open(file, 'r') as f:
            haystack_pile += f.read()
    if not haystack_pile:
        raise ValueError(f"No haystack files found in {haystack_dir}")
    return haystack_pile


def generate_context(haystack_pile_tokens, tokenizer, max_length, needle, depth_percent):
    needle_tokens = tokenizer.encode(needle)
    max_haystack_tokens = max_length - len(needle_tokens)
    start_offset = random.randint(0, len(haystack_pile_tokens) - max_haystack_tokens)
    haystack_subset = haystack_pile_tokens[start_offset:start_offset + max_haystack_tokens]
    insertion_point = int((len(haystack_subset) * depth_percent) / 100)
    if depth_percent != 0 and depth_percent != 100:
        insertion_point += random.randint(-10, 10)
        insertion_point = max(0, min(insertion_point, len(haystack_subset)))
    context_tokens = haystack_subset[:insertion_point] + needle_tokens + haystack_subset[insertion_point:]
    return tokenizer.decode(context_tokens)


def construct_prompt(context, question, city, prompt_type):
    if prompt_type == "gemini_original":
        prompt = f"<context>\n{context}\n</context>\n\n{question} Don't give information outside the document or repeat your findings\n\nHere is the magic number from the context:"
    elif prompt_type == "gemini_simplified":
        prompt = f"<context>\n{context}\n</context>\n\n{question}\n\nHere is the magic number from the context:"
    elif prompt_type == "verbatim":
        prompt = f"<context>\n{context}\n</context>\n\n{question}\n\nthe special magic {city} number is:"
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return prompt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate samples with different prompt types.")
    parser.add_argument(
        '--prompt_type',
        type=str,
        choices=['gemini_original', 'gemini_simplified', 'verbatim'],
        default='verbatim',
        help='Type of prompt to generate.'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=1000,
        help='Number of samples to generate per depth percent.'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=1950,
        help='Maximum length of the context.'
    )
    parser.add_argument(
        '--haystack_dir',
        type=str,
        default='needle/PaulGrahamEssays',
        help='Directory containing haystack text files (relative to PROJECT_ROOT if relative).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='verbatim',
        help='Directory to save the generated samples.'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    depth_percents = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    generate_samples(
        n=args.n,
        max_length=args.max_length,
        haystack_dir=args.haystack_dir,
        tokenizer=tokenizer,
        depth_percents=depth_percents,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type
    )
