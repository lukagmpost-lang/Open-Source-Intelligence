#!/usr/bin/env python3
"""
Build sentiment lexicon for Russian words using LLM classification.

Outputs two sets: POSITIVE and NEGATIVE words.
"""

from osi.answer import call_llm

# Hardcoded list of ~140 Russian sentiment-related words
WORDS = [
    "люблю", "любим", "любит", "любишь", "нравится", "нравлюсь", "спасибо", "благодарю",
    "молодец", "умница", "класс", "классно", "отлично", "супер", "прекрасно", "прекрасный",
    "замечательно", "здорово", "рад", "рада", "счастлив", "счастлива", "хорошо", "хороший",
    "добрый", "милый", "мило", "красиво", "красивый", "обожаю", "целую", "обнимаю",
    "дорогой", "дорогая", "родной", "родная", "улыбка", "улыбаюсь", "весело", "смешно",
    "дружба", "друг", "подруга", "лучший", "лучшая", "ура", "поздравляю", "согласен",
    "согласна", "поддерживаю", "правильно", "верно", "точно", "мир", "свет", "добро",
    "тепло", "нежно", "нежность", "забота", "забочусь",
    "ненавижу", "ненавидим", "ненавидит", "противно", "отвратительно", "ужасно", "ужасный",
    "кошмар", "катастрофа", "плохо", "плохой", "дурак", "дура", "идиот", "идиотка",
    "дебил", "дебилы", "тупой", "тупая", "глупый", "глупая", "злой", "злая", "злость",
    "агрессия", "ненависть", "враг", "враги", "предатель", "предательство", "ложь",
    "врешь", "врут", "обман", "обманул", "обманула", "ужас", "больно", "боль", "грустно",
    "грустный", "печально", "печальный", "плачу", "плакать", "слезы", "страх", "боюсь",
    "боится", "паника", "истерика", "стресс", "депрессия", "устал", "устала", "надоело",
    "достало", "хватит", "заткнись", "молчи", "козел", "козлы", "сволочь", "мразь", "тварь"
]

# Note: I noticed a typo in the original list: "сволочь" should be "сволочь".
# I'll keep it as given but we can fix if needed. For now, use as given.

def main():
    positive = set()
    negative = set()
    batch_size = 20
    total_batches = (len(WORDS) + batch_size - 1) // batch_size
    for i in range(0, len(WORDS), batch_size):
        batch = WORDS[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} words)")
        # Create a numbered list for the prompt
        numbered_list = "\n".join(f"{idx+1}. {word}" for idx, word in enumerate(batch))
        prompt = (
            "Classify each Russian word as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Reply with one line per word in the format '<index> <LABEL>', where index is the 1-based position in the list.\n"
            f"Words:\n{numbered_list}"
        )
        try:
            # Use temperature=0.0 for deterministic output
            response = call_llm(prompt, temperature=0.0)
            # Parse response lines
            for line in response.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    # Skip malformed lines
                    continue
                idx_str, label = parts
                try:
                    idx = int(idx_str) - 1  # convert to 0-based
                except ValueError:
                    continue
                if idx < 0 or idx >= len(batch):
                    continue
                word = batch[idx]
                label_upper = label.upper()
                if label_upper == "POSITIVE":
                    positive.add(word)
                elif label_upper == "NEGATIVE":
                    negative.add(word)
                # NEUTRAL or other labels are ignored
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            continue
    # Write output file
    output_path = "/Users/2950639/Desktop/OSI/Open-Source-Intelligence/src/osi/sentiment_lexicon.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('# Auto-generated sentiment lexicon\\n')
        f.write('POSITIVE = {\\n')
        for w in sorted(positive):
            f.write(f'    "{w}",\\n')
        f.write('}\\n\\n')
        f.write('NEGATIVE = {\\n')
        for w in sorted(negative):
            f.write(f'    "{w}",\\n')
        f.write('}\\n')
    print(f"Lexicon written to {output_path}")
    print(f"Positive words: {len(positive)}")
    print(f"Negative words: {len(negative)}")

if __name__ == "__main__":
    main()