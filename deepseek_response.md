PRVI ODGOVOR:
Odličan izbor teme za diplomski rad! Ovo je zapravo problem **Minimalne šetnje koja pokriva sve vrhove** ...


---
NADOLAZEĆI ODGOVOR:
Odlično pitanje! Ovo je klasičan problem u teoriji grafova sa vrlo lepom teorijskom pozadinom.

Problem koji opisujete poznat je pod nekoliko naziva:
*   **"Minimalna šetnja koja pokriva sve čvorove"**
*   **"Problem popravke puteva"** (Route Inspection Problem) – ali **samo za grafe bez čvorova neparnog stepena**
*   **Poseban slučaj problema putnog rukovaoca** (Generalized Traveling Salesman Problem)

Evo detaljnog teorijskog pregleda:

---

### 1. Formalna Definicija Problema

Dat je **povezan graf G(V, E)**, gde je `V` skup čvorova, a `E` skup grana. Cilj je pronaći **zatvorenu šetnju** (koja počinje i završava se u istom čvoru) koja **posećuje svaki čvor bar jednom**, a da pri tome **ukupna dužina šetnje bude minimalna**.

**Ključna razlika u odnosu na problem trgovačkog putnika (TSP):**
*   **TSP:** Šetnja mora da **posećuje svaki čvor **tačno jednom**** (traži se Hamiltonov ciklus).
*   **Naš problem:** Šetnja može da **posećuje čvorove više puta**. Ovo je ključno olakšanje koje problem čini rešivim u polinomijalnom vremenu.

---

### 2. Teorijski Okvir i Redukcija

Ovaj problem se može efikasno rešiti redukcijom na **Problem Kineskog poštara** (Chinese Postman Problem - CPP).

**Problem Kineskog poštara:** Pronađi **najkraću zatvorenu šetnju** koja **prolazi kroz svaku granu grafa bar jednom**.

**Kako se vrši redukcija?**
Da bismo obišli sve **čvorove**, možemo da "primoramo" šetnju da prođe kroz sve **grane** tako što ćemo graf transformisati.

#### Algoritam (Korak po Korak):

1.  **Kreiranje Metričkog zatvorenja:**
    *   Konstruiše se novi potpuni graf **G'(V, E')**.
    *   Čvorovi ostaju isti.
    *   Težina grane između čvorova `u` i `v` u `G'` postavlja se na dužinu **najkraćeg puta** između `u` i `v` u originalnom grafu `G`. Ovo se lako izračunava algoritmima kao što su **Floyd-Warshall** ili `|V|` poziva **Dijkstra**-inog algoritma.

2.  **Rešavanje Problema Trgovačkog Putnika na Metričkom Zatvorenju:**
    *   Sada, u grafu `G'`, tražimo **Hamiltonov ciklus minimalne težine**. Ovaj ciklus će obići sve čvorove tačno jednom, a njegova težina odgovara ukupnoj dužini najkraćih puteva u originalnom grafu.
    *   **Važno:** Iako je opšti TSP NP-težak, ovde se radi o **Metričkom TSP-u** (težine grana zadovoljavaju nejednakost trougla). Za metrički TSP postoje dobri aproksimacioni algoritmi (npr. Christofidesov algoritam sa faktorom 3/2), a za male instance može se naći i tačno rešenje.

3.  **Transformacija Rešenja Nazad u Originalni Graf:**
    *   Pronađeni TSP ciklus u `G'` ne mora nužno biti validna šetnja u `G` (jer neke grane u `G'` mogu da odgovaraju dugim putevima u `G`).
    *   Zato se svaka grana TSP ciklusa u `G'` zamenjuje nizom grana koje čine odgovarajući najkraći put u originalnom grafu `G`.
    *   Rezultat je **zatvorena šetnja u originalnom grafu G koja posećuje svaki čvor bar jednom**.

---

### 3. Vremenska Složenost

*   **Korak 1 (Metričko zatvorenje):** `O(|V|³)` pomoću Floyd-Warshall-a ili `O(|V| * |E| log |V|)` pomoću `|V|` poziva Dijkstre (korisno za retke grafove).
*   **Korak 2 (TSP na metričkom zatvorenju):**
    *   **Tačno rešenje:** Eksponencijalno `O(|V|² * 2^{|V|})` pomoću dinamičkog programiranje (Hold-Karp). Prihvatljivo za `|V|` do ~20-25.
    *   **Aproksimacija:** Christofidesov algoritam je `O(|V|³)` i daje rešenje unutar 3/2 optimuma.
*   **Korak 3 (Transformacija nazad):** `O(|V| * |E|)` u najgorem slučaju.

**Zaključak:** Ceo problem je **NP-lak** (težak kao metrički TSP), ali se može efikasno aproksimirati za velike instance.

---

### 4. Dodatna Razmatranja

*   **Ako je graf Eulerov:** Ako graf već ima Eulerov ciklus (svi čvorovi parnog stepena), onda je rešenje upravo taj ciklus, jer on prolazi kroz sve grane, a time obilazi i sve čvorove.
*   **Ako je graf Hamiltonov:** Ako graf ima Hamiltonov ciklus, onda je to i optimalno rešenje našeg problema.
*   **Specijalni slučaj - Stabla:** Ako je graf stablo, minimalna šetnja je dvostruki obilazak stabla u dubinu (DFS), ali sa pametnim skraćivanjem. Dužina je `2 * (ukupna težina svih grana) - (dužina najdužeg puta od korena do lista)`.

### 5. Praktična Primena

Ovaj problem ima direktnu primenu u:
*   **Optimizaciji ruta** (npr. dostava poštanskih pošiljki u naselju gde je bitno obići sve ulice, a ne samo raskrsnice).
*   **Planiranju pregleda** (npr. inspekcija svih raskrsnica u gradu).
*   **Projektovanju algoritama za mobilne robote** za istraživanje prostorije.

Za implementaciju, preporučujem korišćenje biblioteka za grafove (npr. `networkx` u Pythonu) koje već imaju implementirane algoritme za najkraći put i TSP.