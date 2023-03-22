# Raktár előrejelző dokumentációja

## A program célja
A program célja a raktárban lévő árukészletek menedzselése, a rendelés mennyiségének t+4 napra prediktálása a boltok várható fogyásának ismeretében
 
## A környezet
A megerősítő tanulás ágense számára létrehoztunk egy környezetet, melyben egy központi raktár, és az alá tartozó boltok találhatóak. A raktár célja, hogy minden nap elég áru legyen, hogy a boltok számára biztosíthassa a másnapi fogyást fedező mennyiséget.

## A környezet pontos felépítése
A raktárban egy vektorban táruljuk a rendelt áru mennyiségét, ezzel szimulálva az időbeli eltolódást a boltok és a raktár között. A vektorba minden nap bekerül a raktár által rendelt mennyiség, valamint kiosztásra kerül a boltok között a négy nappal ezelőtt rendelt összes áru. A boltokban az árut szintén vektorokban tároljuk, ahol az indexek az áru korát (hány napja érkezett a boltba) jelzik. A boltoknál állítható, mennyi idő után kerülnek kidobásra az áruk.


## Tanulás
A tanulás során az ágenst a környezetben szabadjára engedjük, majd az az általa látott információkból (state-space) megpróbálja a lehető legjobb döntést (action) meghozni. A dontés minőségét egy jutalommal (reward) tudjuk kiértékelni.

### State space
Az ágens számára elérhető információ jelenleg:
1. A következő 3 napra jósolt fogyás boltokra bontva
2. A boltok aznapi raktárkészlete
3. Az előző nap kidobásra került áru

### Action
A modellnek ez alapján kell minél pontosabban megrendelnie a 4 nap múlva szükséges árut a boltok számára.

### Reward
A jutalom fő részei:
1. A boltokban kidobásra került áru (negatív)
2. A boltokban lévő, régi áru (negatív)
3. A boltokban a minimum mennyiségen felüli áru (pozitív)

### A tanulás menete
A tanulás célja, hogy megtalálja a megfelelő state-action párokat, azaz a jelen állapot szerint megrendelendő ideális árumennyiséget.
Mivel azonban a tipikus [Q learning](https://en.wikipedia.org/wiki/Q-learning) számára az állapotot leító vektorunk túl nagy, ezért deep q learninget használunk, melyben az egész q tábla bejárása helyett egy neurális hálóval közelítjük az adott állapotban lehetséges actionok q értékeit. Ennek előnye, hogy egyáltalán nincs szükségünk q táblára, viszont a program tanulását, futási idejét jelentősen lassítja. A tanulás során a célunk, hogy a háló paramétereivel az optimális értékekhez konvergáljunk.

#### Epsilon-greedy
A lokális minimumok elkerülése végett egy olyan megoldást kell alkalmazni, mely arra "motiválja" az ágenst, hogy minél több action kipróbáljon. Ez úgy valósul meg, hogy egy epsilon értéktől függ, milyen gyakran választja az általa optimálisnak vélt actiont a modell, egyéb esetben pedig a többi, gyakran még ismeretlen kimenetű actiont fogja megvalósítani. Hogy a modell konvergálni tudjon, az epislon értékét a tanulás során folyamatosan csökkentjük. Természetesen ez a végleges modellnél már nincs jelen.

#### Epizódok
A tanulást megadott hosszú epizódokban folyik, melynek célja, hogy segítse a kapott adatok megértését, a modell jelen állapotának kiértékelését. Minden epizódban összesítődik a napok során szerzett reward, és az adott epizód végén a környezet is visszaállítódik a kezdeti állapotába. Így azonos körülményekkel indulva tesztelhetjük, hogyan változott a teljesítmény a korábbiakhoz képest.

#### Predikciók
A predikciók jelenleg az elkülönült Data classban vannak jelen. Minden bolthoz tartozik egy ilyen objektum, és jelenleg normál eloszlás szerint generál random adatot a boltok, és a raktár számára.

### Tervek
#### State tárolásának átdolgozása
Mivel a reward és az action időben elkülönülnek, ezért lehet, javítana a tanulás minőségén, ha a state-action-reward elmentésekor nem az adott actionhoz tartozó rewardot mentenénk el, hanem azt, mely csak a megrendelt áru kidobásakor keletkezik.

#### Soft-max
Egy alternatív módszer, mely az epsilon-greedyt váltaná fel. Előnye, hogy jobban működik folytonos action space esetén, és nekünk jelen esetben pont ilyen van.
