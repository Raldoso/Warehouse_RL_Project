#Warehouse előrejelző dokumentációja

##A program célja
A program célja a warehouseban lévő storage menedzselése, a rendelés mennyiségének t+4 napra prediktálása.

##A környezet
A megerősítő tanulás ágense számára létrehoztunk egy környezetet, melyben egy központi raktár, és az alá tartozó boltok találhatóak. A raktár célja, hogy minden nap elég áru legyen, hogy a boltok rendeléseit kielégíthesse. A boltok minden nap a másnapi fogyás várható értékére egészítik ki saját árukészletüket.

##A környezet pontos felépítése
A warehouseban egy vektorban táruljuk a rendelt áru mennyiségét, ezzel szimulálva az időbeli eltolódást a boltok és a raktár között. A vektorba minden nap bekerül a warehouse által rendelt mennyiség, valamint kiosztásra kerül a boltok között a négy nappal ezelőtt rendelt összes áru. A boltokban az árut szintén vektorokban tároljuk, ahol az indexek az áru korát (hány napja érkezett a boltba) jelzik.

##Tesztelés
A tesztelés során a boltok eloszlásából húzunk, majd nap végén az adott bolt a másnapi várható fogyasztás értére próbálja meg kiegészíteni saját árukészletét. Ezt a rendelést továbbíta a központi raktár felé, ahol ennek függvényében szétosztásra kerül a korábban az adott napra megrendelt összes áru, a boltok rendelésének nagyságával arányosan. A warehouse a nap végén rendel, az általa rendelt áru csak négy nap múlva kerül a boltokba kiszállításra.

##Hiba 
A hibát a boltokban maradó áru kora jeletni, valamint azon áru mennyisége, melyet meg szerettek volna venni a vásárlók (az eloszlából húzott érték alapján), ám nem volt lehetőségük, mivel a boltban nem volt elég mennyiségű áru. A két hibát egymástól függetlenül kezeljük, így súlyuk szabadon változtatható.

##Predikció
A warehouse rendelésének optimalizálása a fő cél. A predikcióhoz szükséges bemenő paraméterek a boltok várható eloszlásai négy nap múlva, valamint a boltok adott pillanatbeli árukészletei. A modell célje ezen inputok segítségével az optimális rendelendő mennyiség prediktálása.

##További célok
###Vásárlói szokások
Jelen pillanatban a vásárlói szokások abban nyilvánulnak meg, hogy mindig a legfrissebb árut szeretnék megvásárolni, ám később szeretnénk beépíteni, hogy lehetőség legyen ettől eltérő szokásokkal való tesztelére is.
###Lejárati idő
Mivel a boltokban napra pontosan tároljuk, mikor érekezett a termék, így hozzá lehet rendelni minden termékhez egy dátumot, mely után az adott termék mát nem kerülhet eladásra. Ezen termékek számát természetesen a hiba számolásába is be kell építeni.
