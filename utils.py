import pandas as pd
import re
import numpy as np


# =============================================================================
# Patterns regex — définis une seule fois, réutilisés partout
# =============================================================================

# Bot propose une escalade vers un agent humain
PATTERN_ESCALADE = re.compile(
    r"""
    (?:
        agent\s+humain
        | human\s+(?:agent|expert)
        | expert\s+humain
        | human\s+support
        | expert\s+heartist
        | heartist
        | agents?\s+experts?
        | experts?\s+heartist
        | conseiller\s+humain
        | membre\s+de\s+(?:notre\s+)?équipe
        | one\s+of\s+our\s+(?:human\s+)?experts?
        | take\s+a\s+look\s+at\s+this
        | prendre\s+le\s+relais
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bot demande si sa réponse a aidé
PATTERN_FEEDBACK_BOT = re.compile(
    r"""
    (?:
        cela\s+vous\s+a[-\s]?t[-\s]?il\s+aid
        | est[-\s]?ce\s+que\s+cela\s+vous\s+a\s+aid
        | did\s+this\s+help
        | did\s+that\s+help
        | was\s+this\s+helpful
        | did\s+this\s+resolve
        | did\s+this\s+solve
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Feedback positif de l'utilisateur (FR + EN)
PATTERN_FEEDBACK_POSITIF_USER = re.compile(
    r"""
    (
        # FR
        \boui\b
        | merci\b
        | merci\s+beaucoup
        | super\b
        | génial\b
        | parfait\b
        | top\b
        | nickel\b
        | c['']est\s+bon
        | ça\s+marche
        | ca\s+marche
        | c['']est\s+ok
        | tout\s+est\s+bon
        | c['']est\s+parfait
        | impeccable
        | résolu
        | problème\s+résolu

        |

        # EN
        \byes\b
        | \byeah\b
        | \byep\b
        | \bi\s+think\b
        | thanks?\b
        | thank\s+you
        | awesome\b
        | great\b
        | perfect\b
        | amazing\b
        | cool\b
        | nice\b
        | it\s+works
        | works?\s+fine
        | all\s+good
        | resolved\b
        | problem\s+solved
        | that\s+helped
        | this\s+helped
        | ok\b
        | okay\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Feedback négatif de l'utilisateur (FR + EN)
PATTERN_FEEDBACK_NEGATIF_USER = re.compile(
    r"""
    (?:
        \b(?:non|no|nope)\b
        | (?:ça|ca|cela)\s+(?:ne\s+)?(?:m'?a\s+)?pas\s+(?:aidé|aide)
        | (?:not|doesn'?t|didn'?t)\s+(?:help|work|answer)
        | (?:ce\s+n'?est\s+pas|c'?est\s+pas)\s+(?:clair|bon|utile)
        | (?:i\s+still|je\s+ne\s+comprends\s+toujours|je\s+n'?ai\s+toujours)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Acceptation d'une escalade par l'utilisateur
PATTERN_ACCEPTATION_ESCALADE = re.compile(
    r"""
    (
        \b(oui|yes|yep|yeah|ok|okay)\b
        | \b(d['']?accord|volontiers|bien\s+sûr|bien\s+sur)\b
        | \b(je\s+veux\s+bien|allez-y|vas-y|go\s+ahead|please|yes\s+please)\b
        | \b(connectez[-\s]?moi|mettez[-\s]?moi\s+en\s+relation|transférez[-\s]?moi)\b
        | \b(connect\s+me|transfer\s+me|put\s+me\s+through)\b
        | parler\s+à\s+(un\s+)?(expert|agent|humain)
        | talk\s+to\s+(an?\s+)?(agent|human|expert)
        | \b(agent|expert|humain|human)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bot annonce que l'escalade est en cours
PATTERN_ESCALADE_IN_PROGRESS = re.compile(
    r"""
    (
        # FR
        vous\s+serez\s+bient[oô]t\s+
        (?:mis\s+en\s+relation|connect[eé]|transf[eé]r[eé])
        .*?
        (?:expert|agent|humain|heartist|[ée]quipe)

        |

        # EN
        you\s+will\s+soon\s+be\s+
        (?:connected|transferred|put\s+in\s+touch)
        .*?
        (?:expert|agent|human|heartist|support\s+team|team)

        |

        you\s+will\s+be\s+(?:connected|transferred)
        .*?
        (?:expert|agent|human|heartist)

        |

        vous\s+serez\s+
        (?:mis\s+en\s+relation|connect[eé]|transf[eé]r[eé])
        .*?
        (?:expert|agent|humain|heartist)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Requête / question explicite de l'utilisateur
PATTERN_USER_REQUEST = re.compile(
    r"""
    (
        \?
        |
        \b(
            donne[z]?\s+moi
            | dis[\s-]?moi
            | indique[z]?\s+moi
            | explique[z]?\s+moi
            | montre[z]?\s+moi
            | aide[z]?\s+moi
            | peux[\s-]?tu
            | pourrais[\s-]?tu
            | pouvez[\s-]?vous
            | pourriez[\s-]?vous
            | je\s+veux
            | je\s+voudrais
            | j'aimerais
            | je\s+souhaiterais
            | je\s+souhaite
            | je\s+cherche
            | j['']ai\s+besoin
            | besoin\s+d['']aide
            | je\s+n['']arrive\s+pas
            | impossible\s+de
            | marche\s+pas
            | fonctionne\s+pas
            | arrive\s+pas
            | comment
            | pourquoi
            | ou
            | où
            | quand
            | combien
            | que\s+faire
            | quoi\s+faire
            | qu['']est[\s-]?ce\s+que
            | est[\s-]?ce\s+que
            | un\s+problème
            | un\s+souci
            | un\s+pb
            | i\s+need
            | i\s+want
            | i\s+would\s+like
            | i\s+wish
            | can\s+you
            | could\s+you
            | tell\s+me
            | give\s+me
            | show\s+me
            | explain
            | help\s+me
            | how\s+do\s+i
            | how\s+can\s+i
            | what\s+should\s+i
            | why
            | where
            | when
            | issue
            | problem
            | stuck
        )\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bot demande s'il peut aider à autre chose
PATTERN_ANYTHING_ELSE = re.compile(
    r"""
    (
        autre\s+chose
        | autre\s+question
        | autre\s+sujet
        | autre\s+probl[eè]me
        | puis[-\s]?je\s+vous\s+aider
        | est[-\s]?ce\s+que\s+je\s+peux\s+vous\s+aider
        | je\s+peux\s+vous\s+aider
        | je\s+suis\s+l[àa]\s+si\s+besoin
        | n['']h[eé]sitez\s+pas
        | besoin\s+d['']aide
        | vous\s+aider\s+davantage
        | can\s+i\s+help
        | anything\s+else
        | something\s+else
        | do\s+you\s+need\s+help
        | i['']?m\s+here\s+if\s+you\s+need
        | let\s+me\s+know\s+if
        | happy\s+to\s+help
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bot demande une clarification à l'utilisateur
PATTERN_BOT_CLARIFICATION = re.compile(
    r"""
    (
        afin\s+de\s+mieux\s+vous\s+aider
        | pour\s+mieux\s+vous\s+aider
        | pourriez[-\s]?vous
        | pouvez[-\s]?vous
        | j['']?aurais\s+besoin\s+de\s+plus\s+d['']?informations
        | need\s+(?:some\s+)?more\s+information
        | could\s+you\s+(?:please\s+)?(?:provide|clarify|confirm|share|send)
        | can\s+you\s+(?:please\s+)?(?:provide|clarify|confirm|share|send)
        | to\s+better\s+assist\s+you
        | in\s+order\s+to\s+better\s+assist
        | to\s+make\s+sure
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Salutations simples (pas une vraie question)
PATTERN_GREETING_ONLY = re.compile(
    r"""
    ^\s*
    (bonjour|hello|hi|hey|bonsoir|salut|good\s+morning|good\s+evening|oui|non|yes|no)
    [\s,!.?-]*
    (h\d+)?
    [\s,!.?-]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Nouvelle question explicite de l'utilisateur
PATTERN_NEW_QUESTION_EXPLICIT = re.compile(
    r"""
    (
        can\s+i\s+ask
        | i\s+have\s+another\s+question
        | another\s+question
        | one\s+more\s+question
        | i\s+need\s+help\s+with\s+something\s+else
        | another\s+issue
        | different\s+question
        | j['']ai\s+une\s+autre\s+question
        | autre\s+question
        | je\s+voudrais\s+demander\s+autre\s+chose
        | puis[-\s]?je\s+poser\s+une\s+autre\s+question
        | j['']ai\s+un\s+autre\s+sujet
        | autre\s+sujet
        | autre\s+probl[eè]me
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Reformulation / clarification de l'utilisateur (pas une nouvelle question)
PATTERN_USER_CLARIFICATION = re.compile(
    r"""
    (
        # EN
        ^\s*by\s+.+\s+you\s+mean\b
        | ^\s*you\s+mean\b
        | ^\s*it\s+means\b
        | ^\s*that\s+means\b
        | ^\s*so\s+you\s+mean\b
        | ^\s*so\s+it\s+means\b
        | ^\s*do\s+you\s+mean\b
        | ^\s*are\s+you\s+saying\b
        | ^\s*if\s+i\s+understand\s+correctly\b
        | ^\s*if\s+i\s+understand\b
        | ^\s*you\s+are\s+saying\b

        # FR
        | ^\s*tu\s+veux\s+dire\b
        | ^\s*vous\s+voulez\s+dire\b
        | ^\s*ça\s+veut\s+dire\b
        | ^\s*cela\s+veut\s+dire\b
        | ^\s*donc\s+ça\s+veut\s+dire\b
        | ^\s*donc\s+cela\s+veut\s+dire\b
        | ^\s*donc\s+vous\s+voulez\s+dire\b
        | ^\s*donc\s+tu\s+veux\s+dire\b
        | ^\s*si\s+je\s+comprends\b
        | ^\s*si\s+j['']ai\s+bien\s+compris\b
        | ^\s*autrement\s+dit\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Message de continuation (l'utilisateur n'a pas fini)
PATTERN_CONTINUATION = re.compile(
    r"""
    (
        ^\s*je\s+n['']ai\s+pas\s+fini\b
        | ^\s*i\s+(am|was)\s+not\s+finished\b
        | ^\s*wait\b
        | ^\s*attendez\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Message contenant uniquement un lien ou une pièce jointe
PATTERN_ATTACHMENT_OR_LINK = re.compile(
    r"""
    (
        ^\s*attachment\s+\d+\s*:
        | https?://
        | www\.
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Demande explicite de transfert vers un agent ou création de ticket
PATTERN_HANDOFF_OR_TICKET = re.compile(
    r"""
    (
        # FR — ticket
        (créer?|ouvrir|faire|logg?er)\s+(?:un|le|mon|ce)?\s*(?:ticket|dossier)
        |
        # FR — agent / humain / expert
        (parler|échanger|discuter|être\s+mis\s+en\s+relation|me\s+mettre\s+en\s+relation|me\s+connecter)
        \s+(avec\s+)?(un\s+)?(agent|humain|conseiller|expert|heartist)
        |
        # EN — ticket / case
        (open|create|raise|log|submit)\s+(?:a|the|my|this)?\s*(?:ticket|case)
        |
        # EN — agent / human / expert
        (talk|speak|chat|connect(\s+me)?|put\s+me\s+in\s+touch)
        \s+(to|with)?\s*(a\s+)?(human|agent|advisor|expert|heartist)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Demande de mise à jour sur un sujet existant
PATTERN_UPDATE_REQUEST = re.compile(
    r"""
    (
        # FR
        update
        | nouvelles?
        | statut
        | avancement
        | suivi
        | retour
        | des?\s+nouvelles
        | où\s+en\s+est
        | qu['']en\s+est[- ]il
        | vérifier\s+(mon\s+)?(ticket|cas|dossier)
        | concernant\s+(mon\s+)?(ticket|cas|dossier)

        |

        # EN
        any\s+update
        | status
        | follow[- ]?up
        | progress
        | news\s+about
        | check\s+(my\s+)?(ticket|case)
        | update\s+on
        | regarding\s+(my\s+)?(ticket|case)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Question de suivi après annonce d'escalade
PATTERN_ESCALADE_FOLLOWUP = re.compile(
    r"""
    (
        ^\s*(when|quand)\s*\??\s*$
        | do\s+you\s+need
        | can\s+you
        | can\s+i
        | could\s+you
        | should\s+i\s+(?:send|give|provide)
        | before\s+connecting
        | before\s+you\s+connect
        | heartist\s+expert
        | reference\s+(?:first|number)?
        | resaweb
        | avant\s+de\s+(?:me\s+)?mettre\s+en\s+relation
        | avez[-\s]?vous\s+besoin
        | dois[-\s]?je\s+(?:envoyer|donner|fournir)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bot ne trouve pas la réponse
PATTERN_BOT_NO_ANSWER = re.compile(
    r"""
    (?:
        # FR — pas trouvé d'information
        je\s+n['']ai\s+pas\s+trouv[ée]?\s+d['']informations?
        | je\s+n['']ai\s+pas\s+trouv[ée]?\s+d['']informations?\s+spécifiques?
        | je\s+n['']ai\s+pas\s+trouv[ée]?\s+d['']information\s+sur
        | aucune\s+information\s+spécifique
        | information\s+non\s+trouv[ée]e?
        | pas\s+dans\s+(?:ma|notre)\s+(?:base|documentation)

        |

        # EN — pas trouvé d'information
        i\s+could\s+not\s+find\s+information
        | i\s+couldn['']t\s+find\s+information
        | i\s+did\s+not\s+find\s+information
        | i\s+don['']t\s+have\s+information
        | (?:the\s+)?information\s+(?:about|on|regarding).{0,60}(?:is\s+not|not\s+found|unavailable)\s+in\s+my
        | not\s+in\s+my\s+knowledge\s+base
        | is\s+not\s+in\s+my\s+knowledge\s+base

        |

        # EN — bot admet ne pas pouvoir résoudre
        i\s+cannot\s+provide\s+a\s+(?:precise\s+)?(?:solution|answer|response)
        | i\s+am\s+unable\s+to\s+(?:provide|answer|resolve|find)
        | i['']m\s+unable\s+to\s+(?:provide|answer|resolve|find)
        | without\s+(?:seeing|knowing|more\s+information).{0,80}i\s+cannot
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Tentative de résolution / clôture par l'agent humain
PATTERN_AGENT_RESOLUTION = re.compile(
    r"""
    (
        est[- ]ce\s+que\s+(cela|ça)\s+vous\s+a\s+aid
        | est[- ]ce\s+que\s+j['']ai\s+pu\s+vous\s+aider
        | ravi\s+d['']avoir\s+pu\s+vous\s+aider
        | heureux\s+d['']avoir\s+pu\s+vous\s+aider
        | y\s+a[- ]t[- ]il\s+autre\s+chose
        | puis[- ]je\s+faire\s+autre\s+chose
        | ai[- ]je\s+répondu\s+à\s+votre\s+question
        | does\s+this\s+help
        | did\s+this\s+help
        | was\s+i\s+able\s+to\s+help
        | anything\s+else\s+i\s+can\s+help
        | can\s+i\s+help\s+with\s+anything\s+else
        | glad\s+i\s+could\s+help
    )
    """,
    re.I | re.X,
)

# Agent ne peut pas aider
PATTERN_AGENT_FAILURE = re.compile(
    r"""
    (
        pas\s+dans\s+mon\s+p[ée]rim[èe]tre
        | hors\s+de\s+mon\s+scope
        | je\s+ne\s+peux\s+pas
        | impossible\s+de
        | nous\s+ne\s+pouvons\s+pas
        | je\s+n['']ai\s+pas\s+acc[èe]s
        | je\s+ne\s+suis\s+pas\s+en\s+mesure
        | ce\s+n['']est\s+pas\s+possible
        | unfortunately
        | out\s+of\s+scope
        | i\s+cannot
        | i'm\s+unable\s+to
        | not\s+able\s+to
        | no\s+access\s+to
    )
    """,
    re.I | re.X,
)

# Agent envoie un formulaire ou un lien
PATTERN_AGENT_FORM = re.compile(
    r"""
    (
        formulaire | form | survey | questionnaire | request | demande | ticket
        | veuillez\s+remplir | merci\s+de\s+compl[ée]ter
        | fill\s+(in|out) | submit | portal | portail | lien | link
    )
    """,
    re.I | re.X,
)

PATTERN_URL = r"https?://\S+|www\.\S+|\b\S+\.[a-zA-Z]{2,}\S*"

# Identifiant de ticket support (ex: CS12345678)
PATTERN_TICKET_ID = re.compile(r"\bCS\d{8}\b", re.I)


# =============================================================================
# Helpers
# =============================================================================

def classify_speaker(speaker: str) -> str:
    """Classe un locuteur en catégorie : bot, user, human_agent, csat, exclude, other."""
    speaker = str(speaker).lower()
    if "auto-response" in speaker:
        return "exclude"
    if "agent (ai-butler)" in speaker:
        return "bot"
    if "consumer" in speaker:
        return "user"
    if "agent (" in speaker and "butler" not in speaker:
        return "human_agent"
    if "csat" in speaker:
        return "csat"
    return "other"


def get_thumb(messages: pd.Series) -> str:
    """Retourne 'thumbs_up', 'thumbs_down' ou 'none' selon les emojis présents."""
    text = " ".join(messages.astype(str))
    if "👍" in text:
        return "thumbs_up"
    elif "👎" in text:
        return "thumbs_down"
    return "none"


def _flag_bot_state_since_last_user(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Retourne un booléen indiquant si le bot a émis un signal (col=True)
    depuis le dernier message utilisateur, pour chaque ligne.
    Utilisé pour tracker l'état conversationnel entre deux tours user.
    """
    user_msg_counter = (df["speaker_type"] == "user").groupby(df["id"]).cumsum()
    return (
        df.groupby(["id", user_msg_counter])[col]
        .cummax()
        .shift(fill_value=False)
    )


# =============================================================================
# Parsing du transcript brut
# =============================================================================

def get_transcript(conversations: pd.DataFrame) -> pd.DataFrame:
    """
    Parse le champ 'transcript' texte brut en DataFrame de messages individuels.
    Chaque ligne = un message avec son horodatage, locuteur et contenu.
    """
    pattern_msg = (
        r"(?s)\[(.*?)\]\s*([^:\n]+):\s*(.*?)"
        r"(?=\r?\n\[\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+(?:AM|PM)\s+\w+\]|\Z)"
    )

    transcript = (
        conversations
        .assign(messages=conversations["transcript"].str.findall(pattern_msg))
        .explode("messages")
        .dropna(subset=["messages"])
    )

    transcript[["heure_message", "speaker", "message"]] = pd.DataFrame(
        transcript["messages"].tolist(),
        index=transcript.index,
    )

    transcript = (
        transcript[
            (transcript["message"] != "Incoming Chat")
            & (transcript["speaker"] != "Auto-response")
        ]
        .drop(columns=["messages"])
    )

    transcript["heure_message_dt"] = pd.to_datetime(
        transcript["heure_message"]
        .str.replace(" CEST", "", regex=False)
        .str.replace(" CET", "", regex=False),
        format="%m/%d/%Y %I:%M:%S %p",
    )

    transcript = transcript.sort_values(["id", "heure_message_dt"])
    transcript["ordre_message_conv"] = transcript.groupby("id").cumcount() + 1

    return transcript


# =============================================================================
# Segmentation en questions
# =============================================================================

def get_questions(transcript: pd.DataFrame) -> pd.DataFrame:
    """
    Segmente le transcript en questions individuelles.
    Chaque groupe de messages autour d'une requête utilisateur reçoit un question_id.

    Retourne le transcript enrichi avec :
    - speaker_type, question_id, ordre_message_question
    - tous les flags intermédiaires (bot_proposed_escalation, is_new_question_start, etc.)
    """
    df = transcript.copy().sort_values(["id", "ordre_message_conv"])

    df["speaker_type"] = df["speaker"].apply(classify_speaker)
    df["message_clean"] = df["message"].fillna("")

    # --- Premier message utilisateur (hors salutations) ---
    df["user_greeting_only"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_GREETING_ONLY, regex=True)
    )

    first_user_msg = (
        df[(df["speaker_type"] == "user") & ~df["user_greeting_only"]]
        .groupby("id")["ordre_message_conv"]
        .min()
        .reset_index()
        .rename(columns={"ordre_message_conv": "first_user_message"})
    )
    df = df.merge(first_user_msg, how="left", on="id")
    df["has_user_message"] = df["first_user_message"].notna()

    # --- Premier message bot ---
    first_bot_msg = (
        df[df["speaker_type"] == "bot"]
        .groupby("id")["ordre_message_conv"]
        .min()
        .reset_index()
        .rename(columns={"ordre_message_conv": "first_bot_message"})
    )
    df = df.merge(first_bot_msg, how="left", on="id")

    # --- Première intervention agent humain ---
    df["is_human_agent"] = df["speaker_type"] == "human_agent"
    df["human_agent_has_spoken_before"] = (
        df.groupby("id")["is_human_agent"]
        .cummax()
        .shift(1)
        .fillna(False)
    )
    # Flag global : un agent humain a parlé à un moment quelconque dans la conversation
    human_agent_ever_spoke = (
        df.groupby("id")["is_human_agent"]
        .transform("max")
        .astype(bool)
    )
    df["human_agent_ever_spoke"] = human_agent_ever_spoke

    # --- Flags sur les messages utilisateur ---
    df["user_message_is_request"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_USER_REQUEST, na=False)
    )
    df["user_explicit_new_question"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_NEW_QUESTION_EXPLICIT, na=False)
    )
    df["user_clarification_only"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_USER_CLARIFICATION, na=False)
    )
    df["user_continuation_only"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_CONTINUATION, na=False)
    )
    df["user_attachment_or_link_only"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_ATTACHMENT_OR_LINK, na=False)
    )
    df["user_handoff_or_ticket_only"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_HANDOFF_OR_TICKET, na=False)
    )
    df["user_is_update_request"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_UPDATE_REQUEST, na=False)
    )
    df["user_accepts"] = (
        (df["speaker_type"] == "user")
        & df["message_clean"].str.contains(PATTERN_ACCEPTATION_ESCALADE, na=False)
    )

    # --- Flags sur les messages bot ---
    df["bot_asks_anything_else"] = (
        (df["speaker_type"] == "bot")
        & df["message_clean"].str.contains(PATTERN_ANYTHING_ELSE, na=False)
    )
    df["bot_proposed_escalation"] = (
        (df["speaker_type"] == "bot")
        & df["message_clean"].str.contains(PATTERN_ESCALADE, na=False)
        & (df["ordre_message_conv"] > df["first_user_message"])
    )
    df["bot_asked_feedback"] = (
        (df["speaker_type"] == "bot")
        & df["message_clean"].str.contains(PATTERN_FEEDBACK_BOT, na=False)
        & (df["ordre_message_conv"] > df["first_bot_message"])
    )
    df["bot_asks_clarification"] = (
        (df["speaker_type"] == "bot")
        & df["message_clean"].str.contains(PATTERN_BOT_CLARIFICATION, na=False)
    )
    df["bot_escalade_in_progress"] = (
        (df["speaker_type"] == "bot")
        & df["message_clean"].str.contains(PATTERN_ESCALADE_IN_PROGRESS, na=False)
    )

    # --- Contexte : escalade confirmée en cours ---
    escal_confirmed = (
        df[df["bot_escalade_in_progress"]]
        .groupby("id")["ordre_message_conv"]
        .min()
        .reset_index()
        .rename(columns={"ordre_message_conv": "escal_confirmed_msg"})
    )
    df = df.merge(escal_confirmed, how="left", on="id")
    df["escalation_in_process"] = (
        (df["ordre_message_conv"] > df["escal_confirmed_msg"])
        & ~df["human_agent_has_spoken_before"]
    )

    df["user_escalade_followup"] = (
        (df["speaker_type"] == "user")
        & df["escalation_in_process"]
        & df["message_clean"].str.contains(PATTERN_ESCALADE_FOLLOWUP, na=False)
    )

    # --- État du bot depuis le dernier message utilisateur ---
    df["bot_asked_since_last_user"] = (
        _flag_bot_state_since_last_user(df, "bot_asks_anything_else")
        & (df["speaker_type"] == "user")
    )
    df["bot_escalated_since_last_user"] = (
        _flag_bot_state_since_last_user(df, "bot_proposed_escalation")
        & (df["speaker_type"] == "user")
    )
    df["bot_asked_fb_since_last_user"] = (
        _flag_bot_state_since_last_user(df, "bot_asked_feedback")
        & (df["speaker_type"] == "user")
    )
    df["bot_asked_clarification_since_last_user"] = (
        _flag_bot_state_since_last_user(df, "bot_asks_clarification")
        & (df["speaker_type"] == "user")
    )

    df["user_accepts_escalation"] = (
        df["bot_escalated_since_last_user"] & df["user_accepts"]
    )

    # --- Détection des nouvelles questions ---
    # Règle absolue : dès qu'un agent humain est intervenu dans la conversation,
    # on ne crée plus de nouvelle question (l'agent gère la suite).
    df["is_new_question_start"] = (
        (df["speaker_type"] == "user")
        & df["has_user_message"]
        & ~df["user_greeting_only"]
        & ~df["human_agent_ever_spoke"]   # bloque toute Q2+ si un agent est intervenu
        & (
            (
                df["user_explicit_new_question"]
            )
            |
            (
                df["user_message_is_request"]
                & (
                    df["bot_asked_since_last_user"]
                    | df["bot_escalated_since_last_user"]
                    | df["bot_asked_fb_since_last_user"]
                    | (df["ordre_message_conv"] == df["first_user_message"])
                )
                & ~df["bot_asked_clarification_since_last_user"]
                & ~df["user_escalade_followup"]
                & ~df["user_clarification_only"]
                & ~df["user_handoff_or_ticket_only"]
                & ~df["user_continuation_only"]
                & ~df["user_attachment_or_link_only"]
                & ~df["user_accepts_escalation"]
                & ~df["user_is_update_request"]
            )
        )
    )

    # --- Attribution des question_id ---
    df["question_rank"] = df.groupby("id")["is_new_question_start"].cumsum()

    # Messages avant le premier user → rattachés à Q1
    df.loc[(df["has_user_message"]) & (df["question_rank"] == 0), "question_rank"] = 1

    # Conversations sans user message → Q0
    df.loc[~df["has_user_message"], "question_rank"] = 0

    df["question_id"] = np.where(
        df["question_rank"] > 0,
        df["id"].astype(str) + "_Q" + df["question_rank"].astype(int).astype(str),
        np.nan,
    )

    df["ordre_message_question"] = df.groupby(["id", "question_id"]).cumcount() + 1

    return df


# =============================================================================
# Métriques agrégées
# =============================================================================

def conv_gen_metrics(questions: pd.DataFrame) -> pd.DataFrame:
    """Métriques globales par conversation : nb questions, nb messages."""
    return (
        questions.groupby(["id", "customerHandle"])
        .agg(
            nb_questions=("question_id", "nunique"),
            nb_msg_conv=("ordre_message_conv", "max"),
        )
        .reset_index()
    )


def question_gen_metrics(questions: pd.DataFrame) -> pd.DataFrame:
    """Nombre de messages par question."""
    return (
        questions.groupby(["id", "question_id"])
        .agg(nb_msg_question=("ordre_message_question", "max"))
        .reset_index()
    )


# =============================================================================
# Analyses par question
# =============================================================================

def answer_analysis(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte si le bot a répondu à la question et si l'utilisateur
    a exprimé un feedback positif ou négatif après cette réponse.

    Retourne une ligne par (id, question_id) avec :
    - bot_answer : le bot a bien répondu (a demandé un feedback après une réponse)
    - bot_answer_satisfying : l'utilisateur a répondu positivement
    - bot_answer_unsatisfying : l'utilisateur a répondu négativement
    """
    df = questions.copy().sort_values(["id", "ordre_message_question"])

    df["bot_feedback_request"] = (
        (df["speaker_type"] == "bot")
        & df["message"].str.contains(PATTERN_FEEDBACK_BOT, na=False)
    )
    df["bot_no_answer"] = (
        (df["speaker_type"] == "bot")
        & df["message"].str.contains(PATTERN_BOT_NO_ANSWER, na=False)
    )

    # Le bot a demandé un feedback ET avait bien répondu (pas "je n'ai pas trouvé")
    prev_bot_no_answer = df.groupby("id")["bot_no_answer"].shift(1).fillna(False)
    df["bot_feedback_after_answer"] = df["bot_feedback_request"] & ~prev_bot_no_answer

    prev_feedback_request = df.groupby("id")["bot_feedback_request"].shift(1).fillna(False)

    df["feedback_reponse_positive"] = (
        prev_feedback_request
        & (df["speaker_type"] == "user")
        & df["message"].str.contains(PATTERN_FEEDBACK_POSITIF_USER, na=False)
    )
    df["feedback_reponse_negative"] = (
        prev_feedback_request
        & (df["speaker_type"] == "user")
        & df["message"].str.contains(PATTERN_FEEDBACK_NEGATIF_USER, na=False)
    )

    return (
        df.groupby(["id", "question_id"])
        .agg(
            bot_answer=("bot_feedback_after_answer", "max"),
            bot_answer_satisfying=("feedback_reponse_positive", "max"),
            bot_answer_unsatisfying=("feedback_reponse_negative", "max"),
        )
        .reset_index()
    )


def escalation_analysis(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse l'escalade par question :
    - escalation_offered : le bot a proposé un agent humain
    - escalation_accepted : l'utilisateur a accepté
    - escalation_effective : un agent humain a effectivement répondu
    """
    df = questions.copy().sort_values(["id", "ordre_message_conv"])

    df["offers_escalation"] = (
        (df["speaker_type"] == "bot")
        & df["message"].str.contains(PATTERN_ESCALADE, na=False)
        & (df["ordre_message_conv"] > 2)
    )

    prev_escalade = df.groupby("id")["offers_escalation"].shift(1).fillna(False)
    df["acceptation_escalade"] = (
        prev_escalade
        & (df["speaker_type"] == "user")
        & df["message"].str.contains(PATTERN_ACCEPTATION_ESCALADE, na=False)
    )

    df["human_agent"] = df["speaker_type"] == "human_agent"

    return (
        df.groupby(["id", "question_id"])
        .agg(
            escalation_offered=("offers_escalation", "max"),
            escalation_accepted=("acceptation_escalade", "max"),
            escalation_effective=("human_agent", "max"),
        )
        .reset_index()
    )


def agent_reply_analysis(questions: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la réponse de l'agent humain par question.
    Détermine le statut final : ticket_created, redirected_to_form,
    resolved_by_agent, not_resolved, ou unknown.
    """
    df = questions.copy()

    df["is_human_agent"] = df["speaker_type"] == "human_agent"

    df["agent_resolution_attempt"] = (
        df["is_human_agent"]
        & df["message"].str.contains(PATTERN_AGENT_RESOLUTION, na=False)
    )
    df["agent_created_ticket"] = (
        df["is_human_agent"]
        & df["message"].str.contains(PATTERN_TICKET_ID, na=False)
    )
    df["ticket_id"] = (
        df["message"].str.extract(r"\b(CS\d{8})\b", flags=re.I)[0].str.upper()
    )
    df["agent_sent_form"] = (
        df["is_human_agent"]
        & df["message"].str.contains(PATTERN_URL, na=False)
        & df["message"].str.contains(PATTERN_AGENT_FORM, na=False)
    )
    df["agent_could_not_help"] = (
        df["is_human_agent"]
        & df["message"].str.contains(PATTERN_AGENT_FAILURE, na=False)
    )

    agg = (
        df.groupby(["id", "question_id"])
        .agg(
            resolved=("agent_resolution_attempt", "max"),
            ticket=("agent_created_ticket", "max"),
            form=("agent_sent_form", "max"),
            failed=("agent_could_not_help", "max"),
        )
        .reset_index()
    )

    agg["question_status"] = np.select(
        condlist=[agg["ticket"], agg["form"], agg["resolved"], agg["failed"]],
        choicelist=["ticket_created", "redirected_to_form", "resolved_by_agent", "not_resolved"],
        default="unknown",
    )

    return agg


def survey_evaluation(conversations: pd.DataFrame) -> pd.DataFrame:
    """
    Associe le résultat CSAT (thumbs up/down) à chaque conversation.
    Les conversations CSAT sont liées aux vraies conversations via customerHandle + date.

    Retourne une ligne par conversation avec :
    - has_survey : un CSAT a été envoyé ce jour-là pour ce client
    - survey_feedback : 'thumbs_up', 'thumbs_down' ou 'none'
    """
    df = conversations.copy()

    df["is_survey"] = df["assignee"].eq("csat-survey")
    df["date_only"] = (
        pd.to_datetime(df["started"].str.replace(" Europe/Paris", "", regex=False))
        .dt.tz_localize("Europe/Paris")
        .dt.date
    )

    survey_flags = (
        df[df["is_survey"]]
        .groupby(["customerHandle", "date_only"])
        .agg(
            has_survey=("id", "count"),
            survey_feedback=("transcript", get_thumb),
        )
        .reset_index()
    )
    survey_flags["has_survey"] = survey_flags["has_survey"].gt(0)

    result = (
        df[~df["is_survey"]]
        .merge(survey_flags, on=["customerHandle", "date_only"], how="left")
    )

    result["has_survey"] = result["has_survey"].fillna(False)
    result["survey_feedback"] = result["survey_feedback"].fillna(False)

    return result[["id", "has_survey", "survey_feedback"]]


# =============================================================================
# Fonctions utilitaires (moins utilisées)
# =============================================================================

def clean_dates(conversations: pd.DataFrame) -> pd.DataFrame:
    """Extrait et structure les dates de début de conversation."""
    result = conversations[["id", "started"]].copy()
    result["date"] = (
        pd.to_datetime(result["started"].str.replace(" Europe/Paris", "", regex=False))
        .dt.tz_localize("Europe/Paris")
        .dt.date
    )
    result["year"] = pd.to_datetime(result["date"]).dt.year
    result["month"] = pd.to_datetime(result["date"]).dt.month
    return result

HOTEL_CODE_PATTERN = re.compile(r'\b[Hh][A-Za-z0-9]{4}\b')

def _is_valid_hotel_code(value: str) -> bool:
    """Vérifie que la valeur est un code hôtel valide."""
    if not isinstance(value, str):
        return False
    val = value.strip()
    return bool(HOTEL_CODE_PATTERN.fullmatch(val)) and val.upper() != "HXXXX"

def _extract_from_transcript(transcript: str) -> str | None:
    """Cherche un code hôtel dans le transcript."""
    if not isinstance(transcript, str):
        return None
    match = HOTEL_CODE_PATTERN.search(transcript)
    if match:
        code = match.group()
        return code if code.upper() != "HXXXX" else None
    return None

def get_hotel_code(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le code hôtel par conversation.
    
    Priorité :
    1. Colonne 'Hotel Code' si format valide (H/h + 4 alphanum, ≠ 'HXXXX')
    2. Sinon, extraction depuis le transcript
    """
    def resolve_hotel_code(group: pd.DataFrame) -> str | None:
        # 1. Cherche dans la colonne Hotel Code
        for val in group["hotelCode"]:
            if _is_valid_hotel_code(str(val) if pd.notna(val) else ""):
                return val.strip()
        
        # 2. Fallback : cherche dans le transcript
        for transcript in group["Transcript"]:
            code = _extract_from_transcript(transcript)
            if code:
                return code
        
        return None

    result = (
        df.groupby("id")
        .apply(resolve_hotel_code, include_groups=False)
        .reset_index()
    )
    result.columns = ["id", "hotel_code"]
    return result


def get_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le topic par conversation."""
    return (
        df.groupby("id")["Topic"]
        .first()
        .reset_index()
        .rename(columns={"Topic": "topic", "Conversation ID": "id"})
    )


def is_blank(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les questions BLANK."""
    return df[df["Question"] == "BLANK"]
