# Catálogo de emojis y emoticonos

El archivo editable es [emojis_sentimiento.tsv](emojis_sentimiento.tsv).
Se puede abrir en Excel como texto UTF-8 separado por tabulaciones.
Contiene los 5,244 símbolos de `emoji.EMOJI_DATA` en la biblioteca
`emoji==2.16.0`, incluyendo variantes de presentación, tonos de piel y
secuencias compuestas. También incluye 66 emoticonos de texto, como
`:-)`, `:'(`, `T_T` y `^_^`. La cobertura queda ligada a esa versión.

| Columna | Significado |
|---|---|
| simbolo | Secuencia exacta que reconoce el programa |
| tipo | emoji o emoticono |
| nombre | Nombre inglés proporcionado por la biblioteca, o descripción del emoticono |
| polaridad | positive, negative o sin_asignar |
| puntaje | 1, -1 o vacío, respectivamente |
| criterio | heuristica_manual o sin_polaridad_estable |
| version_emoji | Versión Emoji de incorporación indicada por la biblioteca |

## Asignaciones de sentimiento

| Tipo | Positivas | Negativas | Sin asignar | Total |
|---|---:|---:|---:|---:|
| Emoji | 154 | 66 | 5,024 | 5,244 |
| Emoticono | 30 | 27 | 9 | 66 |
| Total | 184 | 93 | 5,033 | 5,310 |

Las polaridades son anotaciones heurísticas de esta ampliación. Se asignan
según usos afectivos habituales y se heredan a las variantes de tono de piel
y presentación. Requieren interpretación contextual. Por ejemplo, 💀 puede
expresar risa, 🔥 puede describir un problema, y 🙃 puede indicar sarcasmo.
La lista incluye objetos, banderas y familias cuyo valor afectivo queda
sin asignar. Esa categoría expresa desconocimiento de la polaridad.

Los nombres y el inventario provienen de los datos distribuidos por la
biblioteca emoji, cuyo paquete incluye su licencia BSD. Las asignaciones
afectivas y los emoticonos fueron redactados para este proyecto. Las reglas
son explícitas en el TSV y se pueden revisar sin modificar Python.

## Uso en el pipeline

`cargar_emojis()` lee y valida el documento. Detecta símbolos duplicados,
tipos y polaridades inválidos y puntajes inconsistentes. El detector procesa
secuencias Unicode completas y conserva cada ocurrencia repetida. Los
emoticonos se buscan desde el más largo para evitar que `>:-(` se cuente
como una coincidencia parcial. Se evitan coincidencias dentro de palabras.

`RasgosSentimiento` genera tres columnas de conteos, en este orden.

1. Símbolos positivos.
2. Símbolos negativos.
3. Símbolos con polaridad sin asignar.

El pipeline escala estos conteos y los concatena con el vector de texto.
La columna puntaje documenta la asignación que determina esos conteos.

Al entrenar, el transformador guarda una copia del catálogo dentro del
modelo serializado. De esta forma, cada modelo usa las mismas asignaciones
con las que fue ajustado.

## Edición y comprobación

1. Editar las filas del TSV con Excel o un editor compatible con UTF-8.
2. Mantener un símbolo único por fila y conservar las siete columnas.
3. Si se cambia una polaridad, actualizar también su puntaje y criterio.
4. Reiniciar el proceso de Python para recargar el documento.
5. Comparar el recurso editado con la referencia sobre dev antes de elegirlo.

La comprobación de esta actualización verifica que las 5,310 entradas se
reconocen individualmente, que las repeticiones se cuentan y que el catálogo
sobrevive a guardar y cargar el transformador. La comparación controlada de
los tres clasificadores está en
[`comparacion_catalogo_emojis_dev.tsv`](../resultados_ampliados/comparacion_catalogo_emojis_dev.tsv).

Los archivos de la corrida de 84 experimentos corresponden al recurso
pequeño anterior. El archivo de comparación registra el hash SHA-256 del
catálogo ampliado y sus resultados en dev. El modelo final publicado usa
únicamente marcado de negación como aumento lingüístico.

En la comparación de dev, el catálogo ampliado empató con las referencias
a seis decimales de macro-F1 en los tres clasificadores.
