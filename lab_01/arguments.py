import sys
import argparse


def main(argv):
    print (argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy z współrzędnymi XYZ")
    parser.add_argument("--plot", help="czy wyświetlać wizualizacje. Domyślnie false.", action="store_true")
    parser.add_argument("--spacing", help="odległość pomiędzy punktami w interpolacji. Domyślnie 1.", type=float)
    parser.add_argument("--min_n_points", help="minimalna liczba punktów do interpolacji komórki. Domyślnie 1.", type=int)
    parser.add_argument("--window_type", help="rodzaj okna do interpolacji. rect lub circle. Domyślnie circle")
    parser.add_argument("--window_size", help="rozmiar okna do wyszukiwania najbliższych punktów. Domyślnie 1.", type=float)
    parser.add_argument("-o", help="plik wynikowy ASCII XYZ. Domyślnie brak zapisu do pliku.")
    parser.add_argument("--method", help="metoda interpolacji. idw, ma lub both. Domyślnie ma (moving_average)")
    parser.add_argument("--idw_exponent", help="wykładnik w metodzie idw. Domyślnie 2")

    args = parser.parse_args()

    if args.i == None:
        print("Podaj nazwę pliku wejściowego: -i <nazwa pliku>")
    if args.spacing == None:
        print("Ustawiam wartość spacing na 1")
        spacing = 1
    else:
        spacing = args.spacing

    if args.min_n_points == None:
        print("Ustawiam wartość min_n_points na 1")
        min_n_points = 1
    else:
        min_n_points = args.min_n_points

    if args.window_type == None:
        print("Ustawiam rodzaj okna na okrąg")
        window_type = "circle"
    else:
        window_type = args.window_type

    if args.window_size == None:
        print("Ustawiam rozmiar okna na 1")
        window_size = 1
    else:
        windows_size = args.windows_size

    if args.method == None:
        print("Ustawiam metodę na ma")
        method = "ma"
    else:
        method = args.method

    idw_exponent = None
    if method == "idw" or method == "both":
        if args.idw_exponent == None:
            print("Ustawiam wykładnik w metodzie idw na 2")
            idw_exponent = 2
        else:
            idw_exponent = args.idw_exponent

