"""Gabarits de prompts (première version ; affinés au jalon 2)."""

from __future__ import annotations

SYSTEM_GENERATOR = (
    "Tu génères des données d'entraînement en {language} pour un petit classifieur. "
    "Tu réponds uniquement avec le JSON demandé."
)

GENERATE_PAIR = """\
Domaine : {domain} — sous-domaine : {subdomain}

Invente une proposition vérifiable du domaine et décris-la par deux descriptions :
- desc_true : ce qui rend la proposition vraie ;
- desc_false : une alternative PRÉCISE qui la rend fausse (pas « tout ce qui n'est pas vrai »).

Puis écris {n_contexts} contextes courts (une à trois phrases, moins de 100 mots) :
- {n_true} de kind "true" (le contexte établit desc_true) ;
- {n_false} de kind "false" (le contexte établit desc_false) ;
- {n_counterfactual} de kind "counterfactual" : édition minimale d'un contexte "true" qui
  inverse la réponse ; indique dans source_index l'index (0-based) du contexte édité ;
- {n_off_topic} de kind "off_topic" : du même domaine mais sans rapport avec les deux descriptions.

Exigences :
- insiste sur la négation, la double négation, la portée de la négation, les quantificateurs
  et les distracteurs plausibles ;
- varie longueur et style : les contextes vrais ne doivent pas être systématiquement plus
  longs ni plus formels que les faux ;
- les contextes ne doivent jamais recopier mot pour mot une description.

Ne réutilise aucune de ces propositions déjà générées : {avoid}
"""

SYSTEM_JUDGE = (
    "Tu es un annotateur rigoureux. Tu estimes une probabilité calibrée. "
    "Tu réponds uniquement avec le JSON demandé."
)

JUDGE = """\
Contexte : « {context} »

Description du VRAI : « {desc_true} »
Description du FAUX : « {desc_false} »

1. off_topic : true si le contexte ne concerne ni l'une ni l'autre description.
2. p_true : probabilité, entre 0 et 1, que le contexte corresponde au VRAI plutôt qu'au FAUX.
   Utilise 0.5 si le contexte est hors sujet ou réellement ambigu.
3. rationale : une phrase de justification.
"""

GENERATE_UNLABELED = """\
Domaine : {domain} — sous-domaine : {subdomain}

Écris {n} textes courts et variés (une à trois phrases) du domaine, en {language} :
messages, notes, extraits d'e-mails, comptes rendus. Inclus des négations, des quantités,
des dates et des montants. Ne réutilise pas ces thèmes : {avoid}
"""

ADVERSARIAL = """\
Domaine : {domain} — sous-domaine : {subdomain}

Description du VRAI : « {desc_true} »
Description du FAUX : « {desc_false} »

Un petit modèle s'est trompé avec assurance sur ces contextes :
{errors}

Écris {n} nouveaux contextes qui piégeraient ce modèle de la même façon (négation, portée,
distracteurs, reformulations), avec pour chacun son kind ("true", "false",
"counterfactual" ou "off_topic").
"""
