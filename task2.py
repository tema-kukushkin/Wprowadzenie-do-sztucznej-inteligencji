import numpy as np
from cec2017.functions import f3, f19


def inicjalizuj_populacje(rozmiar_populacji, wymiar, zakres):
    return np.random.uniform(zakres[0], zakres[1], (rozmiar_populacji, wymiar))


def ocena_dopasowania(populacja, funkcja_celu):
    return np.array([funkcja_celu(ind.reshape(1, -1)) for ind in populacja])


def selekcja_turniejowa(populacja, oceny, rozmiar_turnieju):
    nowa_populacja = []
    for _ in range(len(populacja)):
        uczestnicy = np.random.choice(len(populacja), rozmiar_turnieju, replace=False)
        zwyciezca = uczestnicy[np.argmin(oceny[uczestnicy])]
        nowa_populacja.append(populacja[zwyciezca])
    return np.array(nowa_populacja)


def krzyzowanie_jednopunktowe(rodzic1, rodzic2):
    punkt = np.random.randint(1, len(rodzic1))
    potomek1 = np.concatenate([rodzic1[:punkt], rodzic2[punkt:]])
    potomek2 = np.concatenate([rodzic2[:punkt], rodzic1[punkt:]])
    return potomek1, potomek2


def mutacja(rozwiązanie, prawdopodobieństwo_mutacji, zakres):
    for i in range(len(rozwiązanie)):
        if np.random.rand() < prawdopodobieństwo_mutacji:
            rozwiązanie[i] += np.random.uniform(-0.1, 0.1)
            rozwiązanie[i] = np.clip(rozwiązanie[i], zakres[0], zakres[1])
    return rozwiązanie


def algorytm_ewolucyjny(
    funkcja_celu,
    wymiar,
    zakres,
    rozmiar_populacji=50,
    liczba_pokolen=200,
    rozmiar_turnieju=3,
    prawdopodobieństwo_mutacji=0.1,
):
    populacja = inicjalizuj_populacje(rozmiar_populacji, wymiar, zakres)
    najlepsze_rozwiazanie = None
    najlepsze_dopasowanie = np.inf
    historia_dopasowan = []

    for pokolenie in range(liczba_pokolen):
        oceny = ocena_dopasowania(populacja, funkcja_celu)
        aktualne_najlepsze = np.min(oceny)
        if aktualne_najlepsze < najlepsze_dopasowanie:
            najlepsze_dopasowanie = aktualne_najlepsze
            najlepsze_rozwiazanie = populacja[np.argmin(oceny)]

        nowa_populacja = selekcja_turniejowa(populacja, oceny, rozmiar_turnieju)
        potomkowie = []
        for i in range(0, rozmiar_populacji, 2):
            rodzic1, rodzic2 = nowa_populacja[i], nowa_populacja[i + 1]
            potomek1, potomek2 = krzyzowanie_jednopunktowe(rodzic1, rodzic2)
            potomkowie.append(mutacja(potomek1, prawdopodobieństwo_mutacji, zakres))
            potomkowie.append(mutacja(potomek2, prawdopodobieństwo_mutacji, zakres))
        populacja = np.array(potomkowie)

        historia_dopasowan.append(
            np.min(oceny)
        )  # Dodaj wartość najlepszego rozwiązania w tym pokoleniu

    return najlepsze_rozwiazanie, najlepsze_dopasowanie, historia_dopasowan


# Parametry
wymiar = 10
zakres = (-100, 100)
rozmiar_populacji = 50  # Zmniejszona liczba osobników
liczba_pokolen = 200  # Zmniejszona liczba pokoleń
liczba_powtorzen = 10  # Zmniejszona liczba powtórzeń

# Wyniki z wielu uruchomień
średnie_wyniki_F3 = []
średnie_wyniki_F19 = []

for _ in range(liczba_powtorzen):
    najlepsze_F3, dopasowanie_F3, historia_F3 = algorytm_ewolucyjny(
        f3, wymiar, zakres, rozmiar_populacji, liczba_pokolen
    )
    najlepsze_F19, dopasowanie_F19, historia_F19 = algorytm_ewolucyjny(
        f19, wymiar, zakres, rozmiar_populacji, liczba_pokolen
    )

    średnie_wyniki_F3.append(dopasowanie_F3)
    średnie_wyniki_F19.append(dopasowanie_F19)

# Obliczanie średnich wartości funkcji celu
średnia_F3 = np.mean(średnie_wyniki_F3)
średnia_F19 = np.mean(średnie_wyniki_F19)

print(f"Średnia wartość funkcji dla F3: {średnia_F3}")
print(f"Średnia wartość funkcji dla F19: {średnia_F19}")
