import os
import music21

from pathlib import Path
from music21 import corpus

COMPOSERS = ["bach", "schubert"]

def download_composers(composer_name, folder, limit=20):
    os.makedirs(folder, exist_ok=True)
    print(f"Preuzimanje zavrsenih kompozicija od {composer_name}")

    pieces = corpus.getComposer(composer_name)
    print(f"Nadjeno {len(pieces)} kompozicija od {composer_name}")

    for composition_pointer in pieces[:limit]:
        try:
            original_name = Path(composition_pointer).stem
            file_name = f"{original_name}.mid"
            full_path = os.path.join(folder, file_name)

            if os.path.exists(full_path):
                print(f"Preskocena kompozicija posto vec postoji preuzeta: {file_name}")
                continue

            composition = corpus.parse(composition_pointer)
            composition.write('midi', fp=full_path)

            print(f"Sacuvano: {file_name}")
        except Exception as e:
            print(f"Greska prilikom preuzimanja: {e}")

    print(f"Zavrseno preuzimanje kompozicija od {composer_name}")
    print()

if __name__ == "__main__":
    for composer_name in COMPOSERS:
        download_composers(composer_name=composer_name, folder=f"data/{composer_name}/completed")