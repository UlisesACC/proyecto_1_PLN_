"""Normalización textual y variantes controladas de ortografía y repetición."""
import re
from functools import lru_cache

from nltk import pos_tag, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker

from preprocesamiento import (
    CONTRACCIONES, NEGADORES, RE_ALFANUM, STOPWORDS,
    _lematizador, _pos_wordnet, limpiar, marcar_negacion,
)


LETRAS = re.compile(r"([a-z])\1{2,}")
PALABRA_SIMPLE = re.compile(r"[a-z]{4,24}")
PROTEGIDAS = frozenset("""
    steam valve dota portal counter-strike counterstrike skyrim elder scrolls bethesda
    terraria re-logic stardew minecraft roblox csgo cs2 fps pvp pve rpg mmorpg npc
    nerf nerfed buff buffed grind grinding grindy roguelike roguelite multiplayer
    gameplay matchmaking cheater cheaters smurf smurfing noob noobs gg dlc mod mods
    modding laggy iaa lmao lmfa lol rofl bruh ngl tbh ez op
""".split())


@lru_cache(maxsize=1)
def diccionario_ortografico():
    return SpellChecker(language="en", distance=1)


@lru_cache(maxsize=20000)
def corregir(palabra):
    if not PALABRA_SIMPLE.fullmatch(palabra) or palabra in PROTEGIDAS | NEGADORES:
        return palabra
    diccionario = diccionario_ortografico()
    if palabra in diccionario:
        return palabra
    candidatos = diccionario.candidates(palabra) or ()
    # Desempate alfabético para evitar depender del orden de un conjunto.
    return min(candidatos, key=lambda p: (-diccionario[p], p), default=palabra)


def comprimir_letras(token):
    base = token.lower()
    if not base.isalpha() or not LETRAS.search(base):
        return token
    doble = LETRAS.sub(r"\1\1", base)
    simple = LETRAS.sub(r"\1", base)
    diccionario = diccionario_ortografico()
    if doble in diccionario:
        reducido = doble
    elif simple in diccionario:
        reducido = simple
    else:
        reducido = doble
    return reducido.upper() if token.isupper() else reducido.capitalize() if token.istitle() else reducido


def comprimir_palabras(tokens):
    salida = []
    for token in tokens:
        if token.isalpha() and token.lower() not in NEGADORES and salida:
            if token.lower() == salida[-1].lower():
                continue
        salida.append(token)
    return salida


@lru_cache(maxsize=30000)
def normalizar(texto, nivel, mejoras):
    tokens = word_tokenize(texto if nivel == "N0" else limpiar(texto))
    if nivel != "N0":
        tokens = [CONTRACCIONES.get(token, token) for token in tokens]
    if "letras_repetidas" in mejoras:
        tokens = [comprimir_letras(token) for token in tokens]
    if "ortografia" in mejoras:
        tokens = [
            token if nivel == "N0" and token != token.lower() else corregir(token)
            for token in tokens
        ]
    if "palabras_repetidas" in mejoras:
        tokens = comprimir_palabras(tokens)
    if nivel == "N3":
        tokens = [
            _lematizador.lemmatize(token, _pos_wordnet(tag))
            if RE_ALFANUM.search(token) and token not in NEGADORES else token
            for token, tag in pos_tag(tokens)
        ]
    if "negacion" in mejoras:
        tokens = marcar_negacion(tokens)
    if nivel != "N0":
        tokens = [token for token in tokens if RE_ALFANUM.search(token)]
    if nivel in ("N2", "N3"):
        tokens = [token for token in tokens if token.removeprefix("NEG_") not in STOPWORDS]
    return " ".join(tokens)


class NormalizadorAmpliado(TransformerMixin, BaseEstimator):
    def __init__(self, nivel, mejoras):
        self.nivel = nivel
        self.mejoras = mejoras

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalizar(texto, self.nivel, self.mejoras) for texto in X]

