# LIDL_ML_Project

Warehouse előrejelző dokumentációja

A program célja a warehouseban lévő storage menedzselése, a rendelés mennyiségének t+4 napra prediktálása.
Szükséges bemenetei a boltok eloszlásai, valamint az adott nap végén a boltokban maradt áru száma. Ezen adatok ismeretében előrejelzést ad arra vonatkozólag, mennyi árura lesz szükség a boltokban 4 nap múlva.
A  boltokban a raktárkészletet egy vektor tartja számon, melyben lejárat szerint tároljuk az áru mennyiségét, az újonnan érkező áru tehát balról shiftelődik be minden nap.
A központi raktárban jelen pillanatban csak az adott áru mennyiségét tartjuk számon, lejáratát nem.
A tesztelés során a boltok eloszlásából húzunk, majd nap végén az adott bolt a másnapi várható értékre egészíti ki a saját raktárában lévő árukat. Ezután ezt a rendelést visszaküldi a központi raktárnak, amely megpróbálja pontosan a kért árumennyiséget biztosítani minden bolt számára. Amennyiben ez nem lehetséges, mivel a raktárban nincs erre elegendő áru, a raktár a rendelések mennyiségével arányosan osztja szét az összes árut a boltok között.
Az ágens hibáját a boltokban maradt, lejárt szavatosságú áru és a vevők által megvásárolni kívánt, de nem tudott (mivel a boltnak nem volt áruja készleten) áru mennyisége jelenti.
A vásárló szokások jelen pillanatban úgy működnek, hogy a vásárló mindig a legfrissebb árut kívánja megvenni.
A jövőben szeretnénk megvalósítani, hogy a boltokon túl a warehouseban is vektorosan tároljuk a raktárkészletet.
Mivel a program nem konkrét lejárati dátumot tart számon, hanem azt, hogy az áru hány napja van a boltban, ezért könnyen paraméterezhető, hány nap után tekintjük az árut nem eladhatónak.
