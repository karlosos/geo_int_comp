<p align="center">
    <img src="https://i.imgur.com/urV3GwE.png" width="600px" alt="logo"/>
</p>

***


<h4 align="center">A simple toolset for geospatial (hydrologic) data. Implemented interpolation using moving average and IDW, and compression using DCT. Created for Data Science Languages course in 2020.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#development">Development</a>
</p>

## Key Features

* interpolate data with `interpolation.py`
* compress data with `compression.py`

## How To Use

### Interpolation

```
(.venv) PS C:\Dev\jad_2> python .\interpolation.py --help
usage: interpolation.py [-h] [-i I] [--plot] [--spacing SPACING]
                        [--min_n_points MIN_N_POINTS]
                        [--window_type WINDOW_TYPE]
                        [--window_size WINDOW_SIZE] [-o O]
                        [--method METHOD] [--idw_exponent IDW_EXPONENT]
                        [--pickle]

optional arguments:
  -h, --help            show this help message and exit
  -i I                  plik wejściowy z współrzędnymi XYZ
  --plot                czy wyświetlać wizualizacje. Domyślnie false.
  --spacing SPACING     odległość pomiędzy punktami w interpolacji.
                        Domyślnie 1.
  --min_n_points MIN_N_POINTS
                        minimalna liczba punktów do interpolacji
                        komórki. Domyślnie 1.
  --window_type WINDOW_TYPE
                        rodzaj okna do interpolacji. rect lub circle.
                        Domyślnie circle
  --window_size WINDOW_SIZE
                        rozmiar okna do wyszukiwania najbliższych
                        punktów. Domyślnie 1.
  -o O                  plik wynikowy ASCII XYZ. Domyślnie brak zapisu
                        do pliku.
  --method METHOD       metoda interpolacji. idw, ma lub both.
                        Domyślnie ma (moving_average)
  --idw_exponent IDW_EXPONENT
                        wykładnik w metodzie idw. Domyślnie 2
  --pickle              czy zapisać w formie binarnej. Domyślnie false.
```

<p align="center">
    <img src="https://i.imgur.com/6kPuDXM.png" width="600px" alt="logo"/>
</p>

### Compression

```
(.venv) PS C:\Dev\jad_2> python .\compression.py --help
usage: compression.py [-h] [-i I] [--block_size BLOCK_SIZE]
                      [--decompression_acc DECOMPRESSION_ACC] [--zip]

optional arguments:
  -h, --help            show this help message and exit
  -i I                  plik wejściowy z danymi wejściowymi
  --block_size BLOCK_SIZE
                        Rozmiar bloku danych. Domyślnie blok 8x8
  --decompression_acc DECOMPRESSION_ACC
                        Dokładność dekompresji. Liczba w wartości
                        bewzględnej, np. 0.05m, oznacza to, że po
                        dekompresji w żadnym punkcie błąd nie
                        przekroczy tej wartości.
  --zip                 Czy na końcu dodatkowo kompresujemy dane metodą
                        ZIP
```

<p align="center">
    <img src="https://i.imgur.com/zLgHvcI.png" width="600px" alt="logo"/>
</p>

## Development

1. Create virtual environment with `virtualenv .venv`.
2. Activate venv with `.venv\Scripts\activate.bat`.

## Dataset

Data was ignored in this repo as it would take to long to upload it. Sorry. 😥

## Why the hell didn't you refactor this? :angry:

I got my grade and this code won't be used again.

> "Let me stress that it’s these changes that drive the need to perform refactoring. If the code works and doesn’t ever need to change, it’s perfectly fine to leave it alone. It would be nice to improve it, but unless someone needs to understand it, it isn’t causing any real harm. Yet as soon as someone does need to understand how that code works, and struggles to follow it, then you have to do something about it." Martin Fowler - Refactoring
