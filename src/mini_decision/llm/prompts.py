"""Gabarits de prompts (v2 au jalon 4 : longueurs équilibrées, hors-sujet proches)."""

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
- {n_off_topic} de kind "off_topic" : un texte administratif plausible qui ne permet de
  trancher ni pour l'une ni pour l'autre description. Au moins la moitié sont des hors-sujet
  PROCHES : mêmes personnes, même dossier ou même document que la proposition, mais ils
  parlent d'un autre aspect (ex. pour « la facture est payée » : « la facture a été envoyée
  au service comptable mardi »). Les autres viennent d'un autre sous-domaine.

Exigences :
- insiste sur la négation, la double négation, la portée de la négation, les quantificateurs
  et les distracteurs plausibles ;
- longueurs équilibrées : chaque contexte "false" a à peu près la même longueur (à quelques
  mots près) qu'un contexte "true" ; un contrefactuel garde la longueur de sa source. La
  longueur, le ton et la présence d'une négation ne doivent jamais trahir la réponse : des
  contextes vrais contiennent aussi des négations (« rien ne s'oppose au paiement ») ;
- les contextes ne doivent jamais recopier mot pour mot une description.
- mélange les formats : e-mail, SMS, note interne, extrait de compte rendu, message client.

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
des dates et des montants. Lot n°{batch} : varie les situations, interlocuteurs et tons.
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
